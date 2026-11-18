package dev.openvrcardboard.phone

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Wire formats shared with `desktop/openvrcardboard/protocol.py`.
 *
 * All fields are little-endian. See docs/PROTOCOL.md for the byte layout.
 */
object Protocol {
    const val MAGIC_0 = 'O'.code.toByte()
    const val MAGIC_1 = 'V'.code.toByte()
    const val MAGIC_2 = 'R'.code.toByte()
    const val MAGIC_3 = 'C'.code.toByte()
    const val VERSION: Byte = 1

    const val TYPE_STATE = 0x01
    const val TYPE_HAPTIC = 0x02
    const val TYPE_STATUS = 0x03
    const val TYPE_IMU = 0x20
    const val TYPE_TIME_PING = 0x21
    const val TYPE_TIME_PONG = 0x22

    const val HEADER_SIZE = 12
    const val IMU_SIZE = 64
    const val TIME_PING_SIZE = 20
    const val TIME_PONG_SIZE = 28

    const val LAYOUT_MONO = 0
    const val LAYOUT_SBS = 1
    const val LAYOUT_TOP_BOTTOM = 2

    const val CODEC_JPEG = 0
    const val CODEC_HEVC = 1
    const val CODEC_AV1 = 2
    const val CODEC_H264 = 3

    const val VIDEO_FLAG_KEYFRAME = 0x08
    const val VIDEO_FLAG_CONFIG = 0x10

    fun codecName(value: Int): String = when (value and 0x07) {
        CODEC_HEVC -> "hevc"
        CODEC_AV1 -> "av1"
        CODEC_H264 -> "h264"
        else -> "jpeg"
    }

    fun codecFromName(name: String): Int = when (name.lowercase()) {
        "hevc", "h265" -> CODEC_HEVC
        "av1" -> CODEC_AV1
        "h264", "avc" -> CODEC_H264
        else -> CODEC_JPEG
    }

    fun writeHeader(buffer: ByteBuffer, type: Int, sequence: Int) {
        buffer.put(MAGIC_0)
        buffer.put(MAGIC_1)
        buffer.put(MAGIC_2)
        buffer.put(MAGIC_3)
        buffer.put(VERSION)
        buffer.put(type.toByte())
        buffer.putShort(0)
        buffer.putInt(sequence)
    }

    fun imuPacket(
        sequence: Int,
        timestamp: Double,
        quaternion: FloatArray,
        gyroscope: FloatArray,
        acceleration: FloatArray,
        accuracy: Int,
        screenRotation: Int,
    ): ByteArray {
        val buffer = ByteBuffer.allocate(IMU_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        writeHeader(buffer, TYPE_IMU, sequence)
        buffer.putDouble(timestamp)
        for (value in quaternion) buffer.putFloat(value)
        for (value in gyroscope) buffer.putFloat(value)
        for (value in acceleration) buffer.putFloat(value)
        val tail = ((accuracy and 0x03) or ((screenRotation and 0x03) shl 2)).toByte()
        buffer.put(tail)
        buffer.put(0)
        buffer.put(0)
        buffer.put(0)
        return buffer.array()
    }

    fun writeImu(
        buffer: ByteBuffer,
        sequence: Int,
        timestamp: Double,
        quaternion: FloatArray,
        gyroscope: FloatArray,
        acceleration: FloatArray,
        accuracy: Int,
        screenRotation: Int,
    ) {
        buffer.clear()
        writeHeader(buffer, TYPE_IMU, sequence)
        buffer.putDouble(timestamp)
        for (value in quaternion) buffer.putFloat(value)
        for (value in gyroscope) buffer.putFloat(value)
        for (value in acceleration) buffer.putFloat(value)
        val tail = ((accuracy and 0x03) or ((screenRotation and 0x03) shl 2)).toByte()
        buffer.put(tail)
        buffer.put(0)
        buffer.put(0)
        buffer.put(0)
    }

    fun timePingPacket(sequence: Int, clientTime: Double): ByteArray {
        val buffer = ByteBuffer.allocate(TIME_PING_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        writeHeader(buffer, TYPE_TIME_PING, sequence)
        buffer.putDouble(clientTime)
        return buffer.array()
    }

    fun packetType(data: ByteArray, length: Int): Int? {
        if (length < HEADER_SIZE) return null
        if (data[0] != MAGIC_0 || data[1] != MAGIC_1 ||
            data[2] != MAGIC_2 || data[3] != MAGIC_3
        ) {
            return null
        }
        if (data[4] != VERSION) return null
        return data[5].toInt() and 0xFF
    }

    fun clientTime(data: ByteArray, length: Int): Double? {
        if (length < TIME_PONG_SIZE) return null
        val buffer = ByteBuffer.wrap(data, 0, length).order(ByteOrder.LITTLE_ENDIAN)
        buffer.position(12)
        return buffer.double
    }

    fun serverTime(data: ByteArray, length: Int): Double? {
        if (length < TIME_PONG_SIZE) return null
        val buffer = ByteBuffer.wrap(data, 0, length).order(ByteOrder.LITTLE_ENDIAN)
        buffer.position(20)
        return buffer.double
    }
}
