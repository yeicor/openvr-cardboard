package dev.openvrcardboard.phone

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Surface
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.nio.ByteBuffer
import java.util.ArrayDeque
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/**
 * Camera-mode client: captures the rear camera with Camera2, encodes H.264
 * with MediaCodec and uploads Annex-B access units to the desktop.
 *
 * The stream prefers a 1280x720 mode and requests the highest supported rate
 * up to 120 fps, using a constrained high-speed session when the device only
 * exposes 120 fps there. Only the newest frames are kept; if the network
 * falls behind the queue is cleared and a keyframe is requested so the
 * decoder never receives a broken reference chain.
 */
class CameraStreamer(
    private val host: String,
    private val port: Int,
    private val token: String,
    private val listener: Listener,
) {
    interface Listener {
        fun onCameraStatus(message: String)
        fun onCameraError(message: String)
        fun onCameraFormat(width: Int, height: Int, fps: Int)
    }

    private companion object {
        const val TAG = "OpenVRCardboard"
        const val MAX_QUEUE = 24
        const val TARGET_WIDTH = 1280
        const val TARGET_HEIGHT = 720
        const val DEFAULT_BITRATE_MBPS = 40.0
        const val RECONNECT_MS = 2000L
    }

    private data class CameraMode(
        val size: Size,
        val fps: Int,
        val fpsRange: Range<Int>?,
        val highSpeed: Boolean,
    )

    private data class EncoderChoice(
        val name: String,
        val maxFps: Int,
    )

    private data class EncodedFrame(
        val payload: ByteArray,
        val keyframe: Boolean,
        val presentationTimeUs: Long,
    )

    private val running = AtomicBoolean(false)
    private val pipelineStarted = AtomicBoolean(false)
    private val pipelineGeneration = AtomicLong(0)
    // Some devices advertise constrained high-speed modes that the camera HAL
    // cannot actually start (observed as ERROR_CAMERA_DEVICE). Retry with a
    // standard capture session instead of reconnecting in a loop forever.
    private val highSpeedDisabled = AtomicBoolean(false)
    private val disconnecting = AtomicBoolean(false)
    private val mainHandler = Handler(Looper.getMainLooper())
    private var webSocket: WebSocket? = null
    private var client: OkHttpClient? = null
    private var reconnect: Runnable? = null

    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var encoderThread: Thread? = null
    private var senderThread: Thread? = null

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var encoder: MediaCodec? = null
    private var inputSurface: Surface? = null
    private var codecConfig = ByteArray(0)

    private val encoding = AtomicBoolean(false)
    private val queueLock = Object()
    private val sendQueue = ArrayDeque<EncodedFrame>()
    private var droppingUntilKeyframe = false

    private var sequence = 0
    private var encodedFrames = 0L
    private var sentFrames = 0L
    private var droppedFrames = 0L
    private var selectedWidth = 0
    private var selectedHeight = 0
    private var selectedFps = 0
    private var sensorRotation = 0
    private var horizontalFovDeg = 0.0
    private var bitrateMbps = DEFAULT_BITRATE_MBPS
    private var statsAt = 0L
    private var statsFrames = 0L
    private var lastError = ""

    fun start() {
        if (running.getAndSet(true)) return
        statsAt = SystemClock.elapsedRealtime()
        statsFrames = 0
        connect()
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        mainHandler.removeCallbacksAndMessages(null)
        reconnect = null
        disconnecting.set(false)
        stopPipeline()
        val socket = webSocket
        webSocket = null
        try {
            socket?.cancel()
        } catch (_: Exception) {
        }
        val http = client
        client = null
        try {
            http?.dispatcher?.executorService?.shutdown()
            http?.connectionPool?.evictAll()
        } catch (_: Exception) {
        }
        synchronized(queueLock) {
            sendQueue.clear()
            queueLock.notifyAll()
        }
        listener.onCameraStatus("Camera stopped")
    }

    // ------------------------------------------------------------------
    // WebSocket
    // ------------------------------------------------------------------
    private fun buildUrl(): String {
        var base = host.trim().trimEnd('/')
        if (!base.startsWith("ws://") && !base.startsWith("wss://")) {
            val hasPort = base.matches(Regex(".*:\\d+$")) || base.startsWith("[")
            base = if (hasPort) "ws://$base" else "ws://$base:$port"
        }
        if (!base.substringAfter("://").contains('/')) {
            base += "/camera"
        }
        if (token.isBlank()) return base
        val separator = if (base.contains('?')) '&' else '?'
        return "$base${separator}token=${java.net.URLEncoder.encode(token, "UTF-8")}"
    }

    private fun connect() {
        if (!running.get()) return
        reconnect = null
        lastError = ""
        val http = OkHttpClient.Builder()
            .pingInterval(10, TimeUnit.SECONDS)
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
        client = http
        val request = Request.Builder().url(buildUrl()).build()
        webSocket = http.newWebSocket(request, socketListener)
        listener.onCameraStatus("Connecting to $host:$port…")
    }

    private val socketListener = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: okhttp3.Response) {
            Log.i(TAG, "camera websocket connected")
            // The welcome message may carry the configured bitrate. Start the
            // camera promptly even if the desktop is an older build.
            mainHandler.postDelayed({ startPipelineIfNeeded() }, 1500)
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            try {
                val payload = JSONObject(text)
                when (payload.optString("type")) {
                    "camera_welcome" -> {
                        if (payload.has("bitrate_mbps")) {
                            bitrateMbps = payload.optDouble(
                                "bitrate_mbps",
                                DEFAULT_BITRATE_MBPS,
                            )
                        }
                        startPipelineIfNeeded()
                    }
                    "camera_request_keyframe" -> requestKeyframe()
                    "error" -> {
                        lastError = payload.optString("message", "desktop error")
                        listener.onCameraError(lastError)
                    }
                }
            } catch (_: Exception) {
            }
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) = Unit

        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            webSocket.close(1000, null)
        }

        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            if (!running.get()) return
            handleDisconnect("desktop closed the camera stream")
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: okhttp3.Response?) {
            if (!running.get()) return
            handleDisconnect(t.message ?: "camera connection failed")
        }
    }

    private fun handleDisconnect(message: String) {
        if (!running.get()) return
        if (!disconnecting.compareAndSet(false, true)) return
        mainHandler.removeCallbacksAndMessages(null)
        reconnect = null
        stopPipeline()
        val socket = webSocket
        webSocket = null
        try {
            socket?.cancel()
        } catch (_: Exception) {
        }
        val http = client
        client = null
        try {
            http?.dispatcher?.executorService?.shutdown()
            http?.connectionPool?.evictAll()
        } catch (_: Exception) {
        }
        if (!running.get()) return
        if (message != lastError) {
            lastError = message
            listener.onCameraError(message)
        }
        val retry = Runnable {
            reconnect = null
            disconnecting.set(false)
            if (running.get()) connect()
        }
        reconnect = retry
        mainHandler.postDelayed(retry, RECONNECT_MS)
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------
    private fun startPipelineIfNeeded() {
        if (!running.get()) return
        if (!pipelineStarted.compareAndSet(false, true)) return
        try {
            openCamera()
        } catch (exc: Exception) {
            pipelineStarted.set(false)
            reportError("camera setup failed: ${exc.message ?: exc}")
            handleDisconnect("camera setup failed")
        }
    }

    private fun openCamera() {
        // Closing a camera is asynchronous. A generation token lets a late
        // onOpened/onConfigured callback close itself instead of resurrecting
        // a stopped pipeline or leaking the camera after a quick restart.
        val generation = pipelineGeneration.incrementAndGet()
        val context = AppContextHolder.context
        if (context.checkSelfPermission(Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            throw IllegalStateException("camera permission was not granted")
        }
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = manager.cameraIdList.firstOrNull { id ->
            manager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                CameraCharacteristics.LENS_FACING_BACK
        } ?: manager.cameraIdList.firstOrNull()
            ?: throw IllegalStateException("no camera is available")
        val characteristics = manager.getCameraCharacteristics(cameraId)
        sensorRotation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        horizontalFovDeg = horizontalFov(characteristics)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: throw IllegalStateException("camera has no stream configuration")
        val encoderChoice = chooseEncoder(TARGET_WIDTH, TARGET_HEIGHT)
        val mode = chooseMode(
            characteristics,
            map,
            encoderChoice.maxFps,
            allowHighSpeed = !highSpeedDisabled.get(),
        )
        selectedWidth = mode.size.width
        selectedHeight = mode.size.height
        selectedFps = mode.fps
        Log.i(
            TAG,
            "camera mode $selectedWidth×$selectedHeight @ $selectedFps fps " +
                "(highSpeed=${mode.highSpeed}, range=${mode.fpsRange})",
        )
        listener.onCameraFormat(selectedWidth, selectedHeight, selectedFps)

        val (codec, surface) = createEncoder(
            encoderChoice,
            selectedWidth,
            selectedHeight,
            selectedFps,
        )
        encoder = codec
        inputSurface = surface
        codecConfig = ByteArray(0)
        encoding.set(true)
        startEncoderThread(codec)
        startSenderThread()

        val thread = HandlerThread("openvrcardboard-camera").apply { start() }
        cameraThread = thread
        val handler = Handler(thread.looper)
        cameraHandler = handler

        val stateCallback = object : CameraDevice.StateCallback() {
            override fun onOpened(device: CameraDevice) {
                if (!pipelineCurrent(generation)) {
                    device.close()
                    return
                }
                cameraDevice = device
                try {
                    createSession(device, surface, mode, generation)
                } catch (exc: Exception) {
                    failPipeline(mode, "camera session failed: ${exc.message ?: exc}")
                }
            }

            override fun onDisconnected(device: CameraDevice) {
                device.close()
                if (cameraDevice === device) cameraDevice = null
                if (!pipelineCurrent(generation)) return
                reportError("camera disconnected")
                handleDisconnect("camera disconnected")
            }

            override fun onError(device: CameraDevice, error: Int) {
                device.close()
                if (cameraDevice === device) cameraDevice = null
                if (!pipelineCurrent(generation)) return
                failPipeline(mode, "camera error $error")
            }
        }
        manager.openCamera(cameraId, stateCallback, handler)

        try {
            webSocket?.send(
                JSONObject()
                    .put("type", "camera_hello")
                    .put("protocol", 1)
                    .put("name", "${Build.MANUFACTURER} ${Build.MODEL}")
                    .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
                    .put("camera_id", cameraId)
                    .put(
                        "camera_name",
                        "${Build.MANUFACTURER} ${Build.MODEL} rear camera",
                    )
                    .put("width", selectedWidth)
                    .put("height", selectedHeight)
                    .put("fps", selectedFps)
                    .put("codec", "h264")
                    .put("rotation", sensorRotation)
                    .put("hfov_deg", horizontalFovDeg)
                    .put("bitrate_mbps", bitrateMbps)
                    .toString(),
            )
        } catch (_: Exception) {
        }
        listener.onCameraStatus(
            "Camera $cameraId: $selectedWidth×$selectedHeight @ $selectedFps fps " +
                "(${bitrateMbps.toInt()} Mbps, rotation $sensorRotation°)",
        )
    }

    /**
     * Estimate the horizontal field of view from the back camera's focal
     * length and physical sensor width. The desktop uses it to build a pinhole
     * model for marker PnP instead of assuming the webcam's calibration.
     */
    private fun horizontalFov(characteristics: CameraCharacteristics): Double {
        return try {
            val focal = characteristics
                .get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.firstOrNull()
                ?: return 0.0
            val sensorWidth = characteristics
                .get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                ?.width
                ?: return 0.0
            if (focal <= 0.0f || sensorWidth <= 0.0f) return 0.0
            Math.toDegrees(2.0 * kotlin.math.atan(sensorWidth / (2.0 * focal)))
        } catch (_: Exception) {
            0.0
        }
    }

    private fun chooseEncoder(width: Int, height: Int): EncoderChoice {
        var fallback: EncoderChoice? = null
        try {
            val codecs = MediaCodecList(MediaCodecList.REGULAR_CODECS)
            for (info in codecs.codecInfos) {
                if (!info.isEncoder) continue
                if (info.supportedTypes.none {
                        it.equals(MediaFormat.MIMETYPE_VIDEO_AVC, ignoreCase = true)
                    }
                ) {
                    continue
                }
                val capabilities = try {
                    info.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC)
                } catch (_: Exception) {
                    continue
                }
                val video = capabilities.videoCapabilities ?: continue
                val supported = try {
                    video.isSizeSupported(width, height)
                } catch (_: Exception) {
                    false
                }
                if (!supported) continue
                val maxFps = try {
                    video.getSupportedFrameRatesFor(width, height).upper.toInt()
                } catch (_: Exception) {
                    continue
                }.coerceIn(1, 240)
                val choice = EncoderChoice(info.name, maxFps)
                if (fallback == null) fallback = choice
                // Prefer the first encoder the platform ranks highly that can
                // actually sustain a usable frame rate at this size.
                if (maxFps >= 60) return choice
            }
        } catch (_: Exception) {
        }
        return fallback ?: EncoderChoice("", 120)
    }

    private fun createEncoder(
        choice: EncoderChoice,
        width: Int,
        height: Int,
        fps: Int,
    ): Pair<MediaCodec, Surface> {
        var lastError: Exception? = null
        // Some encoders reject CBR. Retrying VBR keeps the stream alive
        // instead of leaving camera mode in a reconnect loop.
        for (cbr in listOf(true, false)) {
            val format = MediaFormat.createVideoFormat(
                MediaFormat.MIMETYPE_VIDEO_AVC,
                width,
                height,
            )
            format.setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface,
            )
            format.setInteger(
                MediaFormat.KEY_BIT_RATE,
                (bitrateMbps * 1_000_000.0).toInt(),
            )
            format.setInteger(MediaFormat.KEY_FRAME_RATE, fps)
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
            if (cbr) {
                try {
                    format.setInteger(
                        MediaFormat.KEY_BITRATE_MODE,
                        MediaCodecInfo.EncoderCapabilities.BITRATE_MODE_CBR,
                    )
                } catch (_: Exception) {
                }
            }
            try {
                format.setInteger(MediaFormat.KEY_OPERATING_RATE, fps)
                format.setInteger(MediaFormat.KEY_LATENCY, 1)
            } catch (_: Exception) {
            }

            val codec = try {
                if (choice.name.isBlank()) {
                    MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC)
                } else {
                    MediaCodec.createByCodecName(choice.name)
                }
            } catch (exc: Exception) {
                lastError = exc
                continue
            }
            try {
                codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
                val surface = codec.createInputSurface()
                codec.start()
                return codec to surface
            } catch (exc: Exception) {
                lastError = exc
                try {
                    codec.release()
                } catch (_: Exception) {
                }
            }
        }
        throw lastError ?: IllegalStateException("no H.264 encoder is available")
    }

    private fun chooseMode(
        characteristics: CameraCharacteristics,
        map: StreamConfigurationMap,
        encoderMaxFps: Int,
        allowHighSpeed: Boolean,
    ): CameraMode {
        val standardSizes = map.getOutputSizes(SurfaceTexture::class.java)?.toList()
            .orEmpty()
        val standardSize = pickSize(standardSizes)
        val standardDuration = if (standardSize != null) {
            map.getOutputMinFrameDuration(SurfaceTexture::class.java, standardSize)
        } else {
            0L
        }
        val cameraMax = if (standardDuration > 0L) {
            (1_000_000_000L / standardDuration).toInt().coerceIn(1, 240)
        } else {
            30
        }
        val standardMax = minOf(cameraMax, encoderMaxFps).coerceAtLeast(1)
        val ranges = characteristics
            .get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)
            ?.toList()
            .orEmpty()
        val standardRange = chooseStandardRange(ranges, standardMax)
        val standardFps = minOf(
            standardMax,
            standardRange?.upper ?: standardMax,
        ).coerceAtLeast(1)
        var best = CameraMode(
            standardSize ?: Size(TARGET_WIDTH, TARGET_HEIGHT),
            standardFps,
            standardRange,
            false,
        )

        val highSizes = map.highSpeedVideoSizes?.toList().orEmpty()
        val highSize = highSizes.firstOrNull {
            it.width == TARGET_WIDTH && it.height == TARGET_HEIGHT
        } ?: pickSize(highSizes)
        if (allowHighSpeed && highSize != null) {
            val highRanges = map.getHighSpeedVideoFpsRangesFor(highSize)?.toList()
                .orEmpty()
            // Prefer a fixed range such as [120, 120] over a variable one
            // such as [30, 120]. Some camera HALs reject the variable range
            // in a constrained high-speed session with ERROR_CAMERA_DEVICE
            // even though it is advertised. Some devices only publish
            // [120, 240]; running that at 240 would double the bitrate and
            // defeat the 720p120 target, so stay standard there.
            val usableHigh = highRanges.filter {
                it.upper <= minOf(120, encoderMaxFps) && it.upper >= 60
            }
            val highRange = usableHigh
                .filter { it.lower == it.upper }
                .maxByOrNull { it.upper }
                ?: usableHigh.maxByOrNull { it.upper }
            if (highRange != null && highRange.upper > best.fps) {
                best = CameraMode(highSize, highRange.upper, highRange, true)
            }
        }
        return best
    }

    private fun pickSize(sizes: List<Size>): Size? {
        if (sizes.isEmpty()) return null
        sizes.firstOrNull { it.width == TARGET_WIDTH && it.height == TARGET_HEIGHT }
            ?.let { return it }
        val under = sizes.filter {
            it.width <= TARGET_WIDTH && it.height <= TARGET_HEIGHT &&
                it.width >= 640 && it.height >= 360
        }
        if (under.isNotEmpty()) {
            return under.maxByOrNull { it.width * it.height }
        }
        return sizes.minByOrNull { abs(it.width - TARGET_WIDTH) + abs(it.height - TARGET_HEIGHT) }
    }

    private fun chooseStandardRange(
        ranges: List<Range<Int>>,
        maxFps: Int,
    ): Range<Int>? {
        val usable = ranges.filter { it.upper <= maxFps && it.upper >= 15 }
        return usable.maxWithOrNull(
            compareBy<Range<Int>> { it.upper }.thenBy { it.upper - it.lower },
        ) ?: ranges.minByOrNull { abs(it.upper - 30) }
    }

    private fun pipelineCurrent(generation: Long): Boolean =
        running.get() && pipelineGeneration.get() == generation

    private fun createSession(
        device: CameraDevice,
        surface: Surface,
        mode: CameraMode,
        generation: Long,
    ) {
        val characteristics = (AppContextHolder.context
            .getSystemService(Context.CAMERA_SERVICE) as CameraManager)
            .getCameraCharacteristics(device.id)
        val callback = object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(session: CameraCaptureSession) {
                if (!pipelineCurrent(generation)) {
                    session.close()
                    return
                }
                captureSession = session
                try {
                    startRepeating(device, session, surface, mode, characteristics)
                } catch (exc: Exception) {
                    failPipeline(mode, "could not start camera: ${exc.message ?: exc}")
                }
            }

            override fun onConfigureFailed(session: CameraCaptureSession) {
                session.close()
                if (!pipelineCurrent(generation)) return
                failPipeline(mode, "camera rejected the capture session")
            }
        }
        if (mode.highSpeed) {
            device.createConstrainedHighSpeedCaptureSession(
                listOf(surface),
                callback,
                cameraHandler,
            )
        } else {
            device.createCaptureSession(listOf(surface), callback, cameraHandler)
        }
    }

    private fun startRepeating(
        device: CameraDevice,
        session: CameraCaptureSession,
        surface: Surface,
        mode: CameraMode,
        characteristics: CameraCharacteristics,
    ) {
        val request = device.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
        request.addTarget(surface)
        request.set(CaptureRequest.CONTROL_MODE, CameraCharacteristics.CONTROL_MODE_AUTO)
        request.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
        request.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
        mode.fpsRange?.let {
            request.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
        }
        val afModes = characteristics
            .get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES)
            ?.toSet()
            .orEmpty()
        when {
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO in afModes ->
                request.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
                )
            CaptureRequest.CONTROL_AF_MODE_AUTO in afModes ->
                request.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_AUTO,
                )
            else ->
                request.set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CaptureRequest.CONTROL_AF_MODE_OFF,
                )
        }
        val stabilization = characteristics
            .get(CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES)
            ?.toSet()
            .orEmpty()
        if (CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF in stabilization) {
            request.set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
        }
        val built = request.build()
        if (mode.highSpeed) {
            val highSpeedSession = session as CameraConstrainedHighSpeedCaptureSession
            highSpeedSession.setRepeatingBurst(
                highSpeedSession.createHighSpeedRequestList(built),
                null,
                cameraHandler,
            )
        } else {
            session.setRepeatingRequest(built, null, cameraHandler)
        }
    }

    // ------------------------------------------------------------------
    // Encoding and upload
    // ------------------------------------------------------------------
    private fun startEncoderThread(codec: MediaCodec) {
        val thread = Thread({ encodeLoop(codec) }, "openvrcardboard-camera-encode")
        thread.priority = Thread.NORM_PRIORITY + 1
        encoderThread = thread
        thread.start()
    }

    private fun encodeLoop(codec: MediaCodec) {
        val info = MediaCodec.BufferInfo()
        while (encoding.get()) {
            val index = try {
                codec.dequeueOutputBuffer(info, 10_000)
            } catch (exc: Exception) {
                if (encoding.get()) reportError("encoder failed: ${exc.message ?: exc}")
                return
            }
            when {
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val format = codec.outputFormat
                    val sps = readCodecBuffer(format.getByteBuffer("csd-0"))
                    val pps = readCodecBuffer(format.getByteBuffer("csd-1"))
                    if (sps.isNotEmpty() || pps.isNotEmpty()) {
                        codecConfig = sps + pps
                    }
                }
                index >= 0 -> {
                    val keyframe =
                        info.flags and MediaCodec.BUFFER_FLAG_KEY_FRAME != 0
                    if (info.size > 0) {
                        val data = ByteArray(info.size)
                        codec.getOutputBuffer(index)?.let { buffer ->
                            buffer.position(info.offset)
                            buffer.limit(info.offset + info.size)
                            buffer.get(data)
                        }
                        if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
                            if (data.isNotEmpty()) codecConfig = data
                        } else if (data.isNotEmpty()) {
                            val payload = if (keyframe && codecConfig.isNotEmpty()) {
                                codecConfig + data
                            } else {
                                data
                            }
                            enqueueFrame(payload, keyframe, info.presentationTimeUs)
                        }
                    }
                    codec.releaseOutputBuffer(index, false)
                }
            }
        }
    }

    private fun readCodecBuffer(buffer: ByteBuffer?): ByteArray {
        if (buffer == null) return ByteArray(0)
        val copy = buffer.duplicate()
        val bytes = ByteArray(copy.remaining())
        copy.get(bytes)
        return bytes
    }

    private fun enqueueFrame(payload: ByteArray, keyframe: Boolean, ptsUs: Long) {
        synchronized(queueLock) {
            if (!running.get() || !encoding.get()) return
            if (droppingUntilKeyframe && !keyframe) {
                droppedFrames += 1
                return
            }
            if (keyframe) droppingUntilKeyframe = false
            if (sendQueue.size >= MAX_QUEUE) {
                sendQueue.clear()
                droppingUntilKeyframe = true
                droppedFrames += 1
                requestKeyframe()
                return
            }
            sendQueue.addLast(EncodedFrame(payload, keyframe, ptsUs))
            queueLock.notifyAll()
        }
        encodedFrames += 1
    }

    private fun startSenderThread() {
        val thread = Thread({
            while (running.get() && encoding.get()) {
                val frame = synchronized(queueLock) {
                    while (sendQueue.isEmpty() && running.get() && encoding.get()) {
                        try {
                            queueLock.wait(200)
                        } catch (_: InterruptedException) {
                            Thread.currentThread().interrupt()
                            return@synchronized null
                        }
                    }
                    sendQueue.pollFirst()
                } ?: continue
                val socket = webSocket ?: continue
                val packet = Protocol.cameraPacket(
                    sequence,
                    frame.presentationTimeUs / 1_000_000.0,
                    selectedWidth,
                    selectedHeight,
                    frame.payload,
                    frame.keyframe,
                )
                sequence = (sequence + 1) and 0x7FFFFFFF
                val sent = try {
                    socket.send(packet.toByteString())
                } catch (_: Exception) {
                    false
                }
                if (sent) {
                    sentFrames += 1
                    statsFrames += 1
                    reportStats()
                } else {
                    synchronized(queueLock) {
                        sendQueue.clear()
                    }
                    break
                }
            }
        }, "openvrcardboard-camera-send")
        senderThread = thread
        thread.start()
    }

    private fun reportStats() {
        val now = SystemClock.elapsedRealtime()
        if (statsAt == 0L) statsAt = now
        val elapsed = now - statsAt
        if (elapsed < 1000L) return
        val fps = statsFrames * 1000.0 / elapsed
        listener.onCameraStatus(
            "Camera %dx%d @ %d fps · rot %d° · sent %.0f fps · %d frames · %d dropped"
                .format(
                    selectedWidth,
                    selectedHeight,
                    selectedFps,
                    sensorRotation,
                    fps,
                    sentFrames,
                    droppedFrames,
                ),
        )
        Log.i(
            TAG,
            "camera stream %.1f fps, encoded=%d sent=%d dropped=%d".format(
                fps,
                encodedFrames,
                sentFrames,
                droppedFrames,
            ),
        )
        statsAt = now
        statsFrames = 0
    }

    private fun requestKeyframe() {
        val codec = encoder ?: return
        try {
            val params = Bundle()
            params.putInt(MediaCodec.PARAMETER_KEY_REQUEST_SYNC_FRAME, 0)
            codec.setParameters(params)
        } catch (_: Exception) {
        }
    }

    private fun stopPipeline() {
        pipelineGeneration.incrementAndGet()
        pipelineStarted.set(false)
        encoding.set(false)
        synchronized(queueLock) {
            sendQueue.clear()
            queueLock.notifyAll()
        }
        val session = captureSession
        captureSession = null
        try {
            session?.stopRepeating()
        } catch (_: Exception) {
        }
        try {
            session?.close()
        } catch (_: Exception) {
        }
        val device = cameraDevice
        cameraDevice = null
        try {
            device?.close()
        } catch (_: Exception) {
        }
        // Let the encoder thread notice encoding=false and leave dequeue before
        // stopping a codec that is not thread-safe against concurrent calls.
        val currentEncoder = encoderThread
        encoderThread = null
        if (currentEncoder != null && currentEncoder !== Thread.currentThread()) {
            try {
                currentEncoder.join(500)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        val codec = encoder
        encoder = null
        try {
            codec?.stop()
        } catch (_: Exception) {
        }
        try {
            codec?.release()
        } catch (_: Exception) {
        }
        try {
            inputSurface?.release()
        } catch (_: Exception) {
        }
        inputSurface = null
        val currentSender = senderThread
        senderThread = null
        if (currentSender != null && currentSender !== Thread.currentThread()) {
            try {
                currentSender.join(500)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        val thread = cameraThread
        cameraThread = null
        cameraHandler = null
        if (thread != null && Looper.myLooper() != thread.looper) {
            try {
                thread.quitSafely()
                thread.join(500)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        } else {
            thread?.quitSafely()
        }
        codecConfig = ByteArray(0)
    }

    private fun failPipeline(mode: CameraMode, message: String) {
        val fallback = mode.highSpeed && highSpeedDisabled.compareAndSet(false, true)
        val text = if (fallback) {
            "$message; retrying with a standard capture session"
        } else {
            message
        }
        reportError(text)
        handleDisconnect(text)
    }

    private fun reportError(message: String) {
        val text = message.trim()
        if (text.isEmpty() || text == lastError) return
        lastError = text
        Log.w(TAG, "camera: $text")
        listener.onCameraError(text)
    }
}
