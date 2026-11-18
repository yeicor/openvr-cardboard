package dev.openvrcardboard.phone

import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Build
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer

/**
 * Low-latency MediaCodec decoder that renders directly into a [Surface].
 *
 * The desktop streams H.264/HEVC as Annex-B access units and AV1 as temporal
 * units. Rendering to a surface avoids the JPEG decode + bitmap upload path,
 * which is the difference between a 60 FPS phone and a 90+ FPS phone. The
 * decoder is deliberately synchronous: it is driven by one dedicated thread,
 * never the OkHttp reader thread, and drops frames until the next keyframe if
 * the codec falls behind.
 */
class HardwareVideoDecoder(
    private val onFrame: () -> Unit,
    private val onError: (String) -> Unit,
) {
    companion object {
        private const val TAG = "OpenVRCardboard"

        fun mimeFor(codec: String): String = when (codec.lowercase()) {
            "hevc", "h265" -> MediaFormat.MIMETYPE_VIDEO_HEVC
            "av1" -> MediaFormat.MIMETYPE_VIDEO_AV1
            "h264", "avc" -> MediaFormat.MIMETYPE_VIDEO_AVC
            else -> ""
        }

        /** Whether this device exposes a decoder for the codec. */
        fun supports(codec: String): Boolean {
            val mime = mimeFor(codec)
            if (mime.isEmpty()) return false
            return try {
                val list = MediaCodecList(MediaCodecList.REGULAR_CODECS)
                list.codecInfos.any { info ->
                    !info.isEncoder && info.supportedTypes.any { it.equals(mime, true) }
                }
            } catch (_: Exception) {
                false
            }
        }

        /**
         * Extract Annex-B parameter sets from the first keyframe. The desktop
         * also sends them in a config message, but an old desktop or a lost
         * control frame can leave us without codec-specific data.
         */
        fun parameterSets(codec: String, data: ByteArray, offset: Int, length: Int): ByteArray {
            if (codec != "hevc" && codec != "h264") return ByteArray(0)
            val wanted = if (codec == "hevc") setOf(32, 33, 34) else setOf(7, 8)
            val output = java.io.ByteArrayOutputStream()
            var index = findStartCode(data, offset, offset + length)
            while (index >= 0) {
                val nal = index + startCodeLength(data, index, offset + length)
                if (nal >= offset + length) break
                val next = findStartCode(data, nal, offset + length)
                val end = if (next < 0) offset + length else next
                val type = if (codec == "hevc") {
                    (data[nal].toInt() shr 1) and 0x3F
                } else {
                    data[nal].toInt() and 0x1F
                }
                if (type in wanted) {
                    output.write(data, index, end - index)
                }
                index = next
            }
            return output.toByteArray()
        }

        private fun findStartCode(data: ByteArray, start: Int, end: Int): Int {
            var index = start
            while (index + 3 <= end) {
                if (data[index].toInt() == 0 && data[index + 1].toInt() == 0) {
                    if (data[index + 2].toInt() == 1) return index
                    if (index + 3 < end && data[index + 2].toInt() == 0 &&
                        data[index + 3].toInt() == 1
                    ) {
                        return index
                    }
                }
                index += 1
            }
            return -1
        }

        private fun startCodeLength(data: ByteArray, index: Int, end: Int): Int {
            return if (index + 3 < end && data[index + 2].toInt() == 0) 4 else 3
        }
    }

    private var codec: MediaCodec? = null
    private var mime = ""
    private var width = 0
    private var height = 0
    private var presentationUs = 0L
    private var waitingForKeyframe = true

    val configured: Boolean
        get() = codec != null

    fun configure(
        codecName: String,
        frameWidth: Int,
        frameHeight: Int,
        csd: ByteArray,
        surface: Surface,
    ): Boolean {
        release()
        val targetMime = mimeFor(codecName)
        if (targetMime.isEmpty()) {
            onError("unsupported codec $codecName")
            return false
        }
        if (!surface.isValid) {
            onError("display surface is not valid")
            return false
        }
        val decoder = try {
            createDecoder(targetMime)
        } catch (error: Exception) {
            onError("no $codecName decoder: ${error.message ?: "unknown"}")
            return false
        }
        val format = MediaFormat.createVideoFormat(
            targetMime,
            frameWidth.coerceAtLeast(2),
            frameHeight.coerceAtLeast(2),
        )
        format.setInteger(
            MediaFormat.KEY_MAX_INPUT_SIZE,
            maxOf(1 shl 20, frameWidth * frameHeight / 4),
        )
        applyLowLatencyHints(format)
        applyCodecSpecificData(format, codecName, csd)
        return try {
            decoder.configure(format, surface, null, 0)
            decoder.start()
            true
        } catch (error: Exception) {
            try {
                decoder.release()
            } catch (_: Exception) {
            }
            onError("$codecName decoder rejected the stream: ${error.message ?: "unknown"}")
            false
        }.also { started ->
            if (started) {
                codec = decoder
                mime = targetMime
                width = frameWidth
                height = frameHeight
                presentationUs = 0L
                waitingForKeyframe = true
                Log.i(
                    TAG,
                    "hardware decoder started: $codecName ${frameWidth}x$frameHeight",
                )
            }
        }
    }

    /**
     * Prefer a hardware-accelerated decoder. Vendor decoders expose
     * low-latency modes and keep the 90 FPS stream off the CPU; the regular
     * default can be a software decoder on devices with unusual codec lists.
     */
    private fun createDecoder(mime: String): MediaCodec {
        val infos = try {
            MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.filter { info ->
                !info.isEncoder && info.supportedTypes.any { it.equals(mime, true) }
            }
        } catch (_: Exception) {
            emptyList()
        }
        // isHardwareAccelerated() only exists on API 29+, and on older
        // releases the codec list is already ordered with hardware first.
        val usable = infos.filter { info ->
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q || info.isHardwareAccelerated
        }
        for (info in usable.ifEmpty { infos }) {
            try {
                return MediaCodec.createByCodecName(info.name)
            } catch (_: Exception) {
                // A codec can be listed but fail to instantiate (for example
                // when another process owns it); try the next candidate.
            }
        }
        return MediaCodec.createDecoderByType(mime)
    }

    /**
     * Ask the codec for realtime scheduling, the stream's frame rate and the
     * platform low-latency mode. Unknown keys are ignored by most devices and
     * must never prevent configuration.
     */
    private fun applyLowLatencyHints(format: MediaFormat) {
        runCatching { format.setInteger(MediaFormat.KEY_PRIORITY, 0) }
        runCatching { format.setInteger(MediaFormat.KEY_OPERATING_RATE, 90) }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            runCatching { format.setInteger(MediaFormat.KEY_LOW_LATENCY, 1) }
        }
    }

    private fun applyCodecSpecificData(format: MediaFormat, codecName: String, csd: ByteArray) {
        if (csd.isEmpty()) return
        val buffer = ByteBuffer.wrap(csd)
        if (codecName == "h264") {
            // Android's AVC decoder expects SPS and PPS separately.
            val sps = java.io.ByteArrayOutputStream()
            val pps = java.io.ByteArrayOutputStream()
            var index = findStartCode(csd, 0, csd.size)
            while (index >= 0) {
                val nal = index + startCodeLength(csd, index, csd.size)
                if (nal >= csd.size) break
                val next = findStartCode(csd, nal, csd.size)
                val end = if (next < 0) csd.size else next
                when (csd[nal].toInt() and 0x1F) {
                    7 -> sps.write(csd, index, end - index)
                    8 -> pps.write(csd, index, end - index)
                }
                index = next
            }
            if (sps.size() > 0) format.setByteBuffer("csd-0", ByteBuffer.wrap(sps.toByteArray()))
            if (pps.size() > 0) format.setByteBuffer("csd-1", ByteBuffer.wrap(pps.toByteArray()))
        } else {
            // HEVC decoders accept VPS/SPS/PPS in one csd-0 buffer. AV1's
            // sequence header is likewise a single OBU.
            format.setByteBuffer("csd-0", buffer)
        }
    }

    /**
     * Submit one access unit. Returns false when the codec could not accept it
     * so the caller can skip to the next keyframe.
     */
    fun queue(
        data: ByteArray,
        offset: Int,
        length: Int,
        keyframe: Boolean,
        timestamp: Double = 0.0,
    ): Boolean {
        val decoder = codec ?: return false
        if (length <= 0) return true
        if (waitingForKeyframe && !keyframe) return false
        val inputIndex = try {
            decoder.dequeueInputBuffer(8_000L)
        } catch (error: Exception) {
            onError("decoder input failed: ${error.message ?: "unknown"}")
            return false
        }
        if (inputIndex < 0) {
            // Never feed a dependent frame after losing its predecessor.
            waitingForKeyframe = true
            return false
        }
        val input = try {
            decoder.getInputBuffer(inputIndex)
        } catch (error: Exception) {
            onError("decoder buffer failed: ${error.message ?: "unknown"}")
            return false
        }
        if (input == null || length > input.remaining()) {
            try {
                decoder.queueInputBuffer(
                    inputIndex,
                    0,
                    0,
                    0L,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                )
            } catch (_: Exception) {
            }
            waitingForKeyframe = true
            return false
        }
        input.clear()
        input.put(data, offset, length)
        val flags = if (keyframe) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
        // Presentation timestamps come from the desktop sender, so decoder
        // reordering and frame pacing follow the stream's own clock instead
        // of the phone's receive-time jitter.  Keep them strictly increasing
        // for codecs that reject duplicate timestamps.
        val senderUs = if (timestamp > 0.0) (timestamp * 1_000_000.0).toLong() else 0L
        presentationUs = maxOf(presentationUs + 1L, senderUs)
        try {
            decoder.queueInputBuffer(inputIndex, 0, length, presentationUs, flags)
        } catch (error: Exception) {
            onError("decoder queue failed: ${error.message ?: "unknown"}")
            waitingForKeyframe = true
            return false
        }
        waitingForKeyframe = false
        drain(decoder)
        return true
    }

    private fun drain(decoder: MediaCodec) {
        val info = MediaCodec.BufferInfo()
        while (true) {
            val index = try {
                decoder.dequeueOutputBuffer(info, 0L)
            } catch (error: Exception) {
                onError("decoder output failed: ${error.message ?: "unknown"}")
                return
            }
            when {
                index == MediaCodec.INFO_TRY_AGAIN_LATER -> return
                index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                index == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                index >= 0 -> {
                    val render = info.size > 0
                    try {
                        decoder.releaseOutputBuffer(index, render)
                    } catch (error: Exception) {
                        onError("decoder render failed: ${error.message ?: "unknown"}")
                        return
                    }
                    if (render) onFrame()
                }
                else -> return
            }
        }
    }

    fun release() {
        val decoder = codec
        codec = null
        mime = ""
        width = 0
        height = 0
        waitingForKeyframe = true
        if (decoder != null) {
            try {
                decoder.stop()
            } catch (_: Exception) {
            }
            try {
                decoder.release()
            } catch (_: Exception) {
            }
            Log.i(TAG, "hardware decoder released")
        }
    }
}
