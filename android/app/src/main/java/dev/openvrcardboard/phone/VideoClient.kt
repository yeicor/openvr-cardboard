package dev.openvrcardboard.phone

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import android.view.Surface
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.ArrayDeque
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Stream description sent by the desktop in `welcome` and `video_config`.
 *
 * Width/height are the dimensions of the complete stereo frame (both eyes),
 * matching the binary header rather than the per-eye values.
 */
data class VideoFormat(
    val codec: String,
    val width: Int,
    val height: Int,
    val fps: Int,
    val layout: Int,
    val csd: ByteArray,
)

/**
 * Receives the desktop stereo stream over WebSocket.
 *
 * JPEG frames are decoded on a single dedicated worker, with only the newest
 * queued frame kept: when the phone falls behind, older frames are dropped
 * rather than piling up latency. Decoded bitmaps are recycled through a small
 * `inBitmap` pool.
 *
 * H.264/HEVC/AV1 frames go to MediaCodec and render directly to the display
 * surface, skipping Bitmap allocation and upload. The decoder drops dependent
 * frames until the next keyframe if it cannot keep up, so a slow phone shows a
 * brief freeze instead of decoding a corrupt reference chain.
 *
 * The client reconnects every two seconds while running and ignores unknown
 * control messages for forward compatibility.
 */
class VideoClient(
    private val host: String,
    private val port: Int,
    private val token: String,
    private val sensorHz: Int,
    private val displayWidth: Int,
    private val displayHeight: Int,
    private val listener: Listener,
) {
    interface Listener {
        fun onVideoStatus(message: String)
        fun onVideoFrame(bitmap: Bitmap, layout: Int)
        fun onVideoFormat(format: VideoFormat)
        fun onHardwareFrame()
        fun onVideoRecenter()
        fun onVideoError(message: String)
    }

    private companion object {
        const val TAG = "OpenVRCardboard"
        const val HEADER_SIZE = 20
        const val MAX_POOL = 3
        // Two frames of slack hides codec scheduling jitter; a deeper queue
        // only adds latency, so late non-keyframes are dropped.
        const val MAX_HARDWARE_QUEUE = 3
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val running = AtomicBoolean(false)
    private var reconnecting = false
    private var webSocket: WebSocket? = null
    private var client: OkHttpClient? = null

    private val decodeLock = Object()
    private var decodeThread: Thread? = null
    private var pendingJpeg: FrameJob? = null
    private var hardwareQueue = ArrayDeque<FrameJob>()
    private var hardwareNeedsKeyframe = true
    private var hardwareReset = false
    private var queuedFrames = 0L
    private var replacedFrames = 0L
    private var decodedFrames = 0L
    private var droppedFrames = 0L
    private var decodeFailures = 0L
    private var inFlight = 0

    private val freeBitmaps = ArrayDeque<Bitmap>()
    private var poolWidth = 0
    private var poolHeight = 0

    private var formatCodec = ""
    private var formatWidth = 0
    private var formatHeight = 0
    private var formatFps = 0
    private var formatLayout = Protocol.LAYOUT_MONO
    private var formatCsd = ByteArray(0)
    private var surface: Surface? = null
    private var fallbackRequested = false

    private val hardwareDecoder = HardwareVideoDecoder(
        onFrame = { onHardwareFrame() },
        onError = { message ->
            Log.w(TAG, "hardware decoder: $message")
            requestFallback(message)
        },
    )
    private var decoderCodec = ""
    private var decoderWidth = 0
    private var decoderHeight = 0

    private var statsAt = 0L
    private var statsFrames = 0L
    private var statsDropped = 0L
    private var statsReplaced = 0L
    private var lastError = ""

    fun start() {
        if (running.getAndSet(true)) return
        statsAt = SystemClock.elapsedRealtime()
        statsFrames = 0
        statsDropped = 0
        statsReplaced = 0
        fallbackRequested = false
        synchronized(decodeLock) {
            pendingJpeg = null
            hardwareQueue.clear()
            hardwareNeedsKeyframe = true
            hardwareReset = true
            decodeThread = Thread({ decodeLoop() }, "openvrcardboard-video-decode").apply {
                priority = Thread.NORM_PRIORITY + 1
                start()
            }
        }
        connect()
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        mainHandler.removeCallbacksAndMessages(null)
        synchronized(decodeLock) {
            pendingJpeg = null
            hardwareQueue.clear()
            decodeLock.notifyAll()
        }
        try {
            webSocket?.close(1000, "stopped")
        } catch (_: Exception) {
        }
        webSocket = null
        try {
            client?.dispatcher?.executorService?.shutdown()
            client?.connectionPool?.evictAll()
        } catch (_: Exception) {
        }
        client = null
        val thread = synchronized(decodeLock) {
            val current = decodeThread
            decodeThread = null
            current
        }
        if (thread != null && thread !== Thread.currentThread()) {
            try {
                thread.join(500)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            }
        }
        synchronized(decodeLock) {
            while (freeBitmaps.isNotEmpty()) {
                freeBitmaps.removeLast().recycle()
            }
            inFlight = 0
            surface = null
        }
        listener.onVideoStatus("Video stopped")
    }

    /**
     * Supply the surface used by the hardware decoder. MainActivity calls this
     * from `surfaceCreated`/`surfaceDestroyed`; the decode thread reconfigures
     * MediaCodec lazily so a surface swap never races a frame.
     */
    fun setSurface(value: Surface?) {
        synchronized(decodeLock) {
            surface = value
            hardwareReset = true
            decodeLock.notifyAll()
        }
    }

    /**
     * Return a decoded bitmap to the pool once the view has finished with it.
     *
     * The pool recycles stale bitmaps when the stream resolution changes or
     * the client is stopped, so callers only need to release every bitmap they
     * receive from [Listener.onVideoFrame].
     */
    fun release(bitmap: Bitmap) {
        var recycle = false
        synchronized(decodeLock) {
            if (inFlight > 0) inFlight -= 1
            if (
                bitmap.isRecycled ||
                !running.get() ||
                bitmap.width != poolWidth ||
                bitmap.height != poolHeight ||
                bitmap.config != Bitmap.Config.RGB_565 ||
                !bitmap.isMutable ||
                freeBitmaps.size >= MAX_POOL
            ) {
                recycle = true
            } else {
                freeBitmaps.addLast(bitmap)
                decodeLock.notifyAll()
            }
        }
        if (recycle) {
            bitmap.recycle()
        }
    }

    private fun buildUrl(): String {
        var base = host.trim().trimEnd('/')
        if (!base.startsWith("ws://") && !base.startsWith("wss://")) {
            // Accept "host", "host:port" and bare IPv6 addresses. A host that
            // already carries a port keeps it instead of gaining a second one.
            val hasPort = base.matches(Regex(".*:\\d+$")) || base.startsWith("[")
            base = if (hasPort) "ws://$base" else "ws://$base:$port"
        }
        if (!base.substringAfter("://").contains('/')) {
            base += "/video"
        }
        if (token.isBlank()) return base
        val separator = if (base.contains('?')) '&' else '?'
        return "$base${separator}token=${java.net.URLEncoder.encode(token, "UTF-8")}"
    }

    private fun connect() {
        if (!running.get()) return
        fallbackRequested = false
        val http = OkHttpClient.Builder()
            .pingInterval(10, TimeUnit.SECONDS)
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()
        client = http
        val request = try {
            Request.Builder().url(buildUrl()).build()
        } catch (error: Exception) {
            lastError = error.message ?: "invalid URL"
            listener.onVideoError("Video URL error: $lastError")
            scheduleReconnect()
            return
        }
        listener.onVideoStatus("Connecting to ${buildUrl()}")
        webSocket = http.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (!running.get()) {
                    webSocket.close(1000, "stopped")
                    return
                }
                lastError = ""
                listener.onVideoStatus("Video connected")
                val hello = JSONObject()
                    .put("type", "hello")
                    .put("protocol", 2)
                    .put("name", android.os.Build.MODEL)
                    .put("width", displayWidth)
                    .put("height", displayHeight)
                    .put("sensor_hz", sensorHz)
                webSocket.send(hello.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                if (!running.get()) return
                try {
                    val message = JSONObject(text)
                    when (message.optString("type")) {
                        "welcome" -> handleWelcome(message)
                        "video_config" -> handleVideoConfig(message)
                        "recenter" -> listener.onVideoRecenter()
                        "error" -> {
                            lastError = message.optString("message", "desktop error")
                            listener.onVideoError(lastError)
                        }
                        else -> Unit
                    }
                } catch (_: Exception) {
                    // Ignore malformed or unknown control messages.
                }
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                if (!running.get()) return
                enqueue(bytes.toByteArray())
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (!running.get()) return
                listener.onVideoStatus("Video closed")
                scheduleReconnect()
            }

            override fun onFailure(webSocket: WebSocket, error: Throwable, response: Response?) {
                if (!running.get()) return
                lastError = error.message ?: error.javaClass.simpleName
                listener.onVideoError("Video link failed: $lastError")
                scheduleReconnect()
            }
        })
    }

    private fun handleWelcome(message: JSONObject) {
        val video = message.optJSONObject("video") ?: return
        val codec = video.optString("codec", video.optString("format", "jpeg")).lowercase()
        val width = video.optInt("width", 0)
        val height = video.optInt("height", 0)
        val fps = video.optInt("fps", 0)
        val layout = layoutValue(video.optString("layout", "sbs"))
        val csd = decodeBase64(video.optString("csd", ""))
        if (width <= 0 || height <= 0) return
        if (codec != "jpeg" && !HardwareVideoDecoder.supports(codec)) {
            // Keep showing the software view and ask the desktop for JPEG. A
            // device without an AV1/HEVC decoder should still work.
            applyFormat("jpeg", width, height, fps, layout, ByteArray(0))
            requestFallback("no $codec decoder on this phone")
            return
        }
        applyFormat(codec, width, height, fps, layout, csd)
        listener.onVideoStatus(
            "Stream ${layoutName(layout)} ${width}x$height ${codec.uppercase()} @ $fps fps",
        )
    }

    private fun handleVideoConfig(message: JSONObject) {
        val codec = message.optString("codec", "jpeg").lowercase()
        val csd = decodeBase64(message.optString("csd", ""))
        var width = 0
        var height = 0
        var fps = 0
        var layout = Protocol.LAYOUT_MONO
        synchronized(decodeLock) {
            if (formatWidth <= 0 || formatHeight <= 0) return
            applyFormatLocked(
                codec,
                formatWidth,
                formatHeight,
                formatFps,
                formatLayout,
                csd,
            )
            width = formatWidth
            height = formatHeight
            fps = formatFps
            layout = formatLayout
        }
        listener.onVideoFormat(
            VideoFormat(codec, width, height, fps, layout, csd),
        )
    }

    private fun applyFormat(
        codec: String,
        width: Int,
        height: Int,
        fps: Int,
        layout: Int,
        csd: ByteArray,
    ) {
        synchronized(decodeLock) {
            applyFormatLocked(codec, width, height, fps, layout, csd)
        }
        listener.onVideoFormat(VideoFormat(codec, width, height, fps, layout, csd))
    }

    private fun applyFormatLocked(
        codec: String,
        width: Int,
        height: Int,
        fps: Int,
        layout: Int,
        csd: ByteArray,
    ) {
        val changed = codec != formatCodec ||
            width != formatWidth ||
            height != formatHeight ||
            !csd.contentEquals(formatCsd)
        formatCodec = codec
        formatWidth = width
        formatHeight = height
        formatFps = fps
        formatLayout = layout
        formatCsd = csd
        if (changed) {
            hardwareReset = true
            hardwareNeedsKeyframe = true
            if (codec == "jpeg") hardwareQueue.clear()
            decodeLock.notifyAll()
        }
    }

    private fun decodeBase64(value: String): ByteArray {
        if (value.isBlank()) return ByteArray(0)
        return try {
            Base64.decode(value, Base64.DEFAULT)
        } catch (_: IllegalArgumentException) {
            ByteArray(0)
        }
    }

    private fun requestFallback(reason: String) {
        if (fallbackRequested) return
        fallbackRequested = true
        Log.w(TAG, "requesting JPEG fallback: $reason")
        try {
            webSocket?.send(
                JSONObject()
                    .put("type", "video_fallback")
                    .put("reason", reason)
                    .toString(),
            )
        } catch (_: Exception) {
        }
        listener.onVideoError("Switching to JPEG: $reason")
    }

    private fun enqueue(data: ByteArray) {
        if (data.size <= HEADER_SIZE) return
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        val kind = buffer.get().toInt() and 0xFF
        if (kind != 1) return
        val layout = buffer.get().toInt() and 0xFF
        val width = buffer.short.toInt() and 0xFFFF
        val height = buffer.short.toInt() and 0xFFFF
        if (width == 0 || height == 0) return
        val flags = buffer.short.toInt() and 0xFFFF
        val sequence = buffer.getInt()
        val timestamp = buffer.getDouble()
        val codec = Protocol.codecName(flags)
        val keyframe = (flags and Protocol.VIDEO_FLAG_KEYFRAME) != 0
        val job = FrameJob(
            data,
            HEADER_SIZE,
            data.size - HEADER_SIZE,
            layout,
            width,
            height,
            codec,
            keyframe,
            sequence,
            timestamp,
        )
        synchronized(decodeLock) {
            if (!running.get()) return
            if (codec == "jpeg") {
                if (pendingJpeg != null) {
                    replacedFrames += 1
                }
                // The ByteString conversion already copied the wire payload, so
                // keep the array and decode in place at the header offset.
                pendingJpeg = job
                queuedFrames += 1
                decodeLock.notifyAll()
                return
            }
            if (hardwareQueue.size >= MAX_HARDWARE_QUEUE) {
                if (!keyframe) {
                    // The codec is behind.  Dropping this non-keyframe keeps
                    // the reference chain intact and stops latency growing;
                    // a keyframe is allowed to jump the queue and reset it.
                    droppedFrames += 1
                    return
                }
                hardwareQueue.clear()
                hardwareNeedsKeyframe = false
                droppedFrames += 1
            }
            if (hardwareNeedsKeyframe && !keyframe) {
                droppedFrames += 1
                return
            }
            hardwareNeedsKeyframe = false
            hardwareQueue.addLast(job)
            queuedFrames += 1
            decodeLock.notifyAll()
        }
    }

    private fun decodeLoop() {
        try {
            while (running.get()) {
                var jpeg: FrameJob? = null
                var hardware: FrameJob? = null
                var reset: Boolean
                synchronized(decodeLock) {
                    while (
                        running.get() &&
                        pendingJpeg == null &&
                        hardwareQueue.isEmpty() &&
                        !hardwareReset
                    ) {
                        try {
                            decodeLock.wait(250)
                        } catch (_: InterruptedException) {
                            Thread.currentThread().interrupt()
                            return
                        }
                    }
                    if (!running.get()) return
                    reset = hardwareReset
                    hardwareReset = false
                    jpeg = pendingJpeg
                    pendingJpeg = null
                    if (jpeg == null && !reset) {
                        hardware = hardwareQueue.pollFirst()
                    }
                }
                // Copy out of the captured locals so Kotlin can smart-cast
                // them; the synchronized block above is a closure.
                val software = jpeg
                val compressed = hardware
                if (reset) {
                    releaseHardwareDecoder()
                    continue
                }
                if (software != null) {
                    decodeAndDeliver(software)
                } else if (compressed != null) {
                    decodeHardware(compressed)
                }
            }
        } finally {
            releaseHardwareDecoder()
        }
    }

    private data class FrameJob(
        val data: ByteArray,
        val offset: Int,
        val size: Int,
        val layout: Int,
        val width: Int,
        val height: Int,
        val codec: String,
        val keyframe: Boolean,
        val sequence: Int,
        val timestamp: Double,
    )

    private fun decodeHardware(job: FrameJob) {
        val currentSurface: Surface?
        val codec: String
        synchronized(decodeLock) {
            currentSurface = surface
            codec = formatCodec
        }
        if (codec == "jpeg" || job.codec != codec) return
        if (currentSurface == null || !currentSurface.isValid) {
            droppedFrames += 1
            return
        }
        if (
            !hardwareDecoder.configured ||
            decoderCodec != job.codec ||
            decoderWidth != job.width ||
            decoderHeight != job.height
        ) {
            releaseHardwareDecoder()
            val csd = if (formatCsd.isNotEmpty()) {
                formatCsd
            } else {
                HardwareVideoDecoder.parameterSets(job.codec, job.data, job.offset, job.size)
            }
            val configured = hardwareDecoder.configure(
                job.codec,
                job.width,
                job.height,
                csd,
                currentSurface,
            )
            if (!configured) {
                droppedFrames += 1
                return
            }
            decoderCodec = job.codec
            decoderWidth = job.width
            decoderHeight = job.height
        }
        if (!hardwareDecoder.queue(
                job.data,
                job.offset,
                job.size,
                job.keyframe,
                job.timestamp,
            )
        ) {
            droppedFrames += 1
        }
    }

    private fun releaseHardwareDecoder() {
        hardwareDecoder.release()
        decoderCodec = ""
        decoderWidth = 0
        decoderHeight = 0
    }

    private fun onHardwareFrame() {
        synchronized(decodeLock) {
            decodedFrames += 1
            statsFrames += 1
        }
        listener.onHardwareFrame()
        reportStats(formatLayout, formatWidth, formatHeight)
    }

    private fun decodeAndDeliver(job: FrameJob) {
        var reusable: Bitmap? = null
        synchronized(decodeLock) {
            if (job.width != poolWidth || job.height != poolHeight) {
                while (freeBitmaps.isNotEmpty()) {
                    freeBitmaps.removeLast().recycle()
                }
                poolWidth = job.width
                poolHeight = job.height
            }
            if (freeBitmaps.isNotEmpty()) {
                reusable = freeBitmaps.removeLast()
            } else if (inFlight >= MAX_POOL) {
                droppedFrames += 1
                return
            }
        }

        val reusableBitmap = reusable
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.RGB_565
            inMutable = true
            reusableBitmap?.let { inBitmap = it }
        }
        val started = SystemClock.elapsedRealtimeNanos()
        var bitmap = decode(job.data, job.offset, options)
        if (bitmap == null && reusableBitmap != null) {
            // A few JPEG variants (for example grayscale) cannot reuse an
            // RGB_565 bitmap. Retry once with a fresh allocation.
            options.inBitmap = null
            bitmap = decode(job.data, job.offset, options)
        }
        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

        if (bitmap == null) {
            synchronized(decodeLock) {
                if (reusableBitmap != null && !reusableBitmap.isRecycled) {
                    freeBitmaps.addLast(reusableBitmap)
                }
                decodeFailures += 1
            }
            return
        }

        synchronized(decodeLock) {
            if (
                reusableBitmap != null &&
                reusableBitmap !== bitmap &&
                !reusableBitmap.isRecycled
            ) {
                // BitmapFactory did not use the provided allocation.
                if (freeBitmaps.size < MAX_POOL) freeBitmaps.addLast(reusableBitmap)
                else reusableBitmap.recycle()
            }
            inFlight += 1
            decodedFrames += 1
            statsFrames += 1
        }
        Log.v(
            TAG,
            "decoded frame ${job.width}x${job.height} in %.2f ms".format(elapsedMs),
        )
        listener.onVideoFrame(bitmap, job.layout)
        reportStats(job.layout, job.width, job.height)
    }

    private fun decode(data: ByteArray, offset: Int, options: BitmapFactory.Options): Bitmap? {
        return try {
            BitmapFactory.decodeByteArray(data, offset, data.size - offset, options)
        } catch (_: IllegalArgumentException) {
            null
        } catch (_: OutOfMemoryError) {
            null
        }
    }

    private fun reportStats(layout: Int, width: Int, height: Int) {
        val now = SystemClock.elapsedRealtime()
        if (statsAt == 0L) {
            statsAt = now
            return
        }
        val elapsed = now - statsAt
        if (elapsed < 2000L) return
        val fps: Double
        val dropped: Long
        val replaced: Long
        val frames: Long
        synchronized(decodeLock) {
            fps = statsFrames * 1000.0 / elapsed
            dropped = droppedFrames - statsDropped
            replaced = replacedFrames - statsReplaced
            frames = decodedFrames
            statsAt = now
            statsFrames = 0
            statsDropped = droppedFrames
            statsReplaced = replacedFrames
        }
        val codec = synchronized(decodeLock) { formatCodec.ifBlank { "jpeg" } }
        listener.onVideoStatus(
            "Video %d frames · %s %s %dx%d · %.1f fps · dropped %d".format(
                frames,
                codec.uppercase(),
                layoutName(layout),
                width,
                height,
                fps,
                dropped + replaced,
            ),
        )
    }

    private fun layoutValue(value: String): Int = when (value.lowercase()) {
        "sbs", "side-by-side", "side_by_side" -> Protocol.LAYOUT_SBS
        "top-bottom", "top_bottom", "tb" -> Protocol.LAYOUT_TOP_BOTTOM
        else -> Protocol.LAYOUT_MONO
    }

    private fun layoutName(layout: Int): String = when (layout) {
        Protocol.LAYOUT_SBS -> "SBS"
        Protocol.LAYOUT_TOP_BOTTOM -> "top-bottom"
        else -> "mono"
    }

    private fun scheduleReconnect() {
        if (!running.get() || reconnecting) return
        reconnecting = true
        mainHandler.postDelayed({
            reconnecting = false
            if (!running.get()) return@postDelayed
            try {
                client?.dispatcher?.executorService?.shutdown()
            } catch (_: Exception) {
            }
            connect()
        }, 2000)
    }
}
