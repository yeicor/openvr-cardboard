package dev.openvrcardboard.phone

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Checks the Kotlin wire encoder against the layouts in docs/PROTOCOL.md and
 * `desktop/openvrcardboard/protocol.py`. These are plain JVM tests: no Android
 * device or emulator is required.
 */
class ProtocolTest {
    private fun wrap(data: ByteArray): ByteBuffer =
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

    private fun loadGoldenVectors(): Map<String, ByteArray> {
        val path = System.getProperty("ovrc.goldenVectors")
            ?: "../../protocol/golden_vectors.txt"
        val vectors = mutableMapOf<String, ByteArray>()
        File(path).forEachLine { line ->
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) return@forEachLine
            val parts = trimmed.split(Regex("\\s+"))
            if (parts.size != 2) return@forEachLine
            val bytes = parts[1].chunked(2).map { it.toInt(16).toByte() }.toByteArray()
            vectors[parts[0]] = bytes
        }
        return vectors
    }

    private fun assertHeader(data: ByteArray, expectedType: Int, expectedSequence: Int) {
        assertEquals('O'.code.toByte(), data[0])
        assertEquals('V'.code.toByte(), data[1])
        assertEquals('R'.code.toByte(), data[2])
        assertEquals('C'.code.toByte(), data[3])
        assertEquals(Protocol.VERSION, data[4])
        assertEquals(expectedType.toByte(), data[5])
        assertEquals(0, data[6].toInt())
        assertEquals(0, data[7].toInt())
        assertEquals(expectedSequence, wrap(data).getInt(8))
    }

    @Test
    fun imuPacketMatchesDocumentedLayout() {
        val packet = Protocol.imuPacket(
            sequence = 0x01020304,
            timestamp = 1234.5,
            quaternion = floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f),
            gyroscope = floatArrayOf(1.5f, -2.5f, 3.5f),
            acceleration = floatArrayOf(0.0f, -9.81f, 0.25f),
            accuracy = 3,
            screenRotation = 2,
        )

        assertEquals(Protocol.IMU_SIZE, packet.size)
        assertHeader(packet, Protocol.TYPE_IMU, 0x01020304)

        val buffer = wrap(packet)
        assertEquals(1234.5, buffer.getDouble(12), 0.0)
        val floats = floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f, 1.5f, -2.5f, 3.5f, 0.0f, -9.81f, 0.25f)
        floats.forEachIndexed { index, expected ->
            assertEquals("float $index", expected, buffer.getFloat(20 + index * 4), 0.0f)
        }

        // Tail byte packs accuracy in bits 0-1 and screen rotation in bits 2-3.
        assertEquals((3 or (2 shl 2)).toByte(), packet[60])
        assertEquals(0, packet[61].toInt())
        assertEquals(0, packet[62].toInt())
        assertEquals(0, packet[63].toInt())
    }

    @Test
    fun imuPacketMasksPackFlags() {
        val packet = Protocol.imuPacket(
            sequence = 1,
            timestamp = 0.0,
            quaternion = FloatArray(4),
            gyroscope = FloatArray(3),
            acceleration = FloatArray(3),
            accuracy = 0x7,
            screenRotation = 0x5,
        )
        // accuracy 0x7 -> 3, rotation 0x5 -> 1, combined 0b0111 = 7.
        assertEquals(7.toByte(), packet[60])
    }

    @Test
    fun writeImuReusesBuffer() {
        val buffer = ByteBuffer.allocate(Protocol.IMU_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        Protocol.writeImu(
            buffer,
            sequence = 42,
            timestamp = 7.25,
            quaternion = floatArrayOf(0f, 0f, 0f, 1f),
            gyroscope = FloatArray(3),
            acceleration = floatArrayOf(0f, 0f, 9.81f),
            accuracy = 2,
            screenRotation = 3,
        )

        val packet = buffer.array()
        assertEquals(Protocol.IMU_SIZE, packet.size)
        assertHeader(packet, Protocol.TYPE_IMU, 42)
        assertEquals(7.25, wrap(packet).getDouble(12), 0.0)
        assertEquals((2 or (3 shl 2)).toByte(), packet[60])
        assertArrayEquals(ByteArray(3), packet.copyOfRange(61, 64))
    }

    @Test
    fun timePingMatchesDocumentedLayout() {
        val packet = Protocol.timePingPacket(sequence = 9, clientTime = 55.125)
        assertEquals(Protocol.TIME_PING_SIZE, packet.size)
        assertHeader(packet, Protocol.TYPE_TIME_PING, 9)
        assertEquals(55.125, wrap(packet).getDouble(12), 0.0)
    }

    @Test
    fun goldenVectorsMatchSharedPythonAndCppFile() {
        val vectors = loadGoldenVectors()
        val auth = Protocol.authTag("test-token")
        assertEquals(Protocol.AUTH_SIZE, auth.size)

        val imu = Protocol.imuPacket(
            sequence = 0x01020304,
            timestamp = 1234.5,
            quaternion = floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f),
            gyroscope = floatArrayOf(1.5f, -2.5f, 3.5f),
            acceleration = floatArrayOf(0.0f, -9.81f, 0.25f),
            accuracy = 3,
            screenRotation = 2,
        )
        assertArrayEquals(vectors.getValue("imu"), imu)
        assertArrayEquals(
            vectors.getValue("imu_auth"),
            Protocol.imuPacket(
                sequence = 0x01020304,
                timestamp = 1234.5,
                quaternion = floatArrayOf(0.1f, 0.2f, 0.3f, 0.9f),
                gyroscope = floatArrayOf(1.5f, -2.5f, 3.5f),
                acceleration = floatArrayOf(0.0f, -9.81f, 0.25f),
                accuracy = 3,
                screenRotation = 2,
                authTag = auth,
            ),
        )

        assertArrayEquals(
            vectors.getValue("time_ping"),
            Protocol.timePingPacket(sequence = 9, clientTime = 55.125),
        )
        assertArrayEquals(
            vectors.getValue("time_ping_auth"),
            Protocol.timePingPacket(sequence = 9, clientTime = 55.125, authTag = auth),
        )
    }

    @Test
    fun packetTypeRejectsBadHeaders() {
        val ping = Protocol.timePingPacket(sequence = 1, clientTime = 0.0)
        assertEquals(Protocol.TYPE_TIME_PING, Protocol.packetType(ping, ping.size))

        assertNull("short packet", Protocol.packetType(ping.copyOf(4), 4))

        val badMagic = ping.copyOf()
        badMagic[0] = 'X'.code.toByte()
        assertNull(Protocol.packetType(badMagic, badMagic.size))

        val badVersion = ping.copyOf()
        badVersion[4] = 2
        assertNull(Protocol.packetType(badVersion, badVersion.size))

        val unknownType = ping.copyOf()
        unknownType[5] = 0x7F
        assertEquals(0x7F, Protocol.packetType(unknownType, unknownType.size))
    }

    @Test
    fun timePongFieldsAreReadAtDocumentedOffsets() {
        val packet = ByteArray(Protocol.TIME_PONG_SIZE)
        val buffer = wrap(packet)
        Protocol.writeHeader(buffer, Protocol.TYPE_TIME_PONG, 3)
        buffer.putDouble(12, 111.5)
        buffer.putDouble(20, 222.75)

        assertEquals(Protocol.TYPE_TIME_PONG, Protocol.packetType(packet, packet.size))
        assertEquals(111.5, Protocol.clientTime(packet, packet.size)!!, 0.0)
        assertEquals(222.75, Protocol.serverTime(packet, packet.size)!!, 0.0)
        assertNull(Protocol.clientTime(packet, 12))
        assertNull(Protocol.serverTime(packet, 20))
    }
}
