package dev.openvrcardboard.phone

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Streams rotation vector, gyroscope and linear acceleration samples over UDP.
 *
 * Packets are sent from the rotation-vector callback (at the requested sensor
 * period) so the latest orientation always wins. A second thread answers the
 * desktop clock-sync pings; the offset is diagnostic but exposed for status.
 */
class ImuSender(
    private val host: String,
    private val port: Int,
    sensorHz: Int,
    private val token: String,
    private val rotationProvider: () -> Int,
    private val listener: Listener,
) : SensorEventListener {

    interface Listener {
        fun onImuStatus(message: String)
        fun onImuError(message: String)
    }

    private val sensorManager: SensorManager
    private val sensorThread: HandlerThread
    private val sensorHandler: Handler
    private val running = AtomicBoolean(false)
    private var sendSequence = 0
    private var socket: DatagramSocket? = null
    private var target: InetSocketAddress? = null
    private var receiveThread: Thread? = null
    private var pingThread: Thread? = null

    private var rotationSensor: Sensor? = null
    private var gyroSensor: Sensor? = null
    private var accelSensor: Sensor? = null

    private val gyroscope = FloatArray(3)
    private val acceleration = FloatArray(3)
    private var accuracy = 3
    private var lastStatusAt = 0L
    private var lastSentTimestampNs = 0L
    private var gyroTimestampNs = 0L
    private var accelTimestampNs = 0L
    private val gyroSample = FloatArray(3)
    private val accelSample = FloatArray(3)

    @Volatile
    var packetsSent: Long = 0L
        private set

    @Volatile
    var lastError: String = ""
        private set

    @Volatile
    var clockOffsetMs: Double = 0.0
        private set

    private val authTag = Protocol.authTag(token)
    private val buffer = ByteBuffer.allocate(Protocol.IMU_SIZE + authTag.size)
        .order(ByteOrder.LITTLE_ENDIAN)
    private val receiveBuffer = ByteArray(128)

    private val periodUs: Int = (1_000_000 / sensorHz.coerceIn(10, 400)).coerceAtLeast(1)

    init {
        val context = AppContextHolder.context
        sensorManager = context.getSystemService(SensorManager::class.java)
        rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        sensorThread = HandlerThread("openvrcardboard-imu").also { it.start() }
        sensorHandler = Handler(sensorThread.looper)
    }

    val clockSource: Int
        get() = if (clockOffsetMs == 0.0) 0 else 1

    fun start() {
        if (running.getAndSet(true)) return
        val rotation = rotationSensor
        if (rotation == null) {
            running.set(false)
            lastError = "No rotation-vector sensor on this phone"
            listener.onImuError(lastError)
            return
        }
        try {
            socket = DatagramSocket().apply { soTimeout = 500 }
            target = InetSocketAddress(InetAddress.getByName(host), port)
        } catch (error: Exception) {
            running.set(false)
            lastError = error.message ?: error.javaClass.simpleName
            listener.onImuError("UDP setup failed: $lastError")
            stopSockets()
            return
        }

        sensorManager.registerListener(this, rotation, periodUs, sensorHandler)
        gyroSensor?.let { sensorManager.registerListener(this, it, periodUs, sensorHandler) }
        accelSensor?.let { sensorManager.registerListener(this, it, periodUs, sensorHandler) }
        startClockSync()
        listener.onImuStatus("IMU ${sensorHz()} Hz → $host:$port")
    }

    fun stop() {
        if (!running.getAndSet(false)) return
        sensorManager.unregisterListener(this)
        stopSockets()
        try {
            sensorThread.quitSafely()
        } catch (_: Exception) {
        }
        listener.onImuStatus("IMU stopped")
    }

    private fun sensorHz(): Int = 1_000_000 / periodUs

    private fun stopSockets() {
        try {
            socket?.close()
        } catch (_: Exception) {
        }
        socket = null
        receiveThread?.interrupt()
        receiveThread = null
        pingThread?.interrupt()
        pingThread = null
    }

    private fun startClockSync() {
        val sock = socket ?: return
        val destination = target ?: return
        pingThread = Thread({
            var pingSequence = 0
            while (running.get() && !Thread.currentThread().isInterrupted) {
                try {
                    val clientTime = SystemClock.elapsedRealtimeNanos() / 1e9
                    val ping = Protocol.timePingPacket(pingSequence++, clientTime, authTag)
                    sock.send(DatagramPacket(ping, ping.size, destination))
                    Thread.sleep(5000)
                } catch (_: InterruptedException) {
                    break
                } catch (error: Exception) {
                    if (!running.get()) break
                    try {
                        Thread.sleep(1000)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }
        }, "openvrcardboard-time-ping").also { it.isDaemon = true; it.start() }

        receiveThread = Thread({
            while (running.get() && !Thread.currentThread().isInterrupted) {
                try {
                    val packet = DatagramPacket(receiveBuffer, receiveBuffer.size)
                    sock.receive(packet)
                    val type = Protocol.packetType(receiveBuffer, packet.length) ?: continue
                    if (type != Protocol.TYPE_TIME_PONG) continue
                    val client = Protocol.clientTime(receiveBuffer, packet.length) ?: continue
                    val server = Protocol.serverTime(receiveBuffer, packet.length) ?: continue
                    val now = SystemClock.elapsedRealtimeNanos() / 1e9
                    val rtt = now - client
                    if (rtt in 0.0..1.0) {
                        clockOffsetMs = (server - (client + rtt / 2.0)) * 1000.0
                    }
                } catch (_: InterruptedException) {
                    break
                } catch (_: java.net.SocketTimeoutException) {
                } catch (error: Exception) {
                    if (!running.get()) break
                }
            }
        }, "openvrcardboard-time-recv").also { it.isDaemon = true; it.start() }
    }

    private fun sendPacket(sensorTimestampNs: Long, orientation: FloatArray) {
        val sock = socket ?: return
        val destination = target ?: return
        if (!running.get()) return
        if (sensorTimestampNs <= lastSentTimestampNs) return
        lastSentTimestampNs = sensorTimestampNs
        // Pair the three sensors by their own event timestamps.  The rotation
        // callback fires first in a sensor batch, so the small post below
        // lets the gyro and accelerometer events for the same instant land
        // before the packet is built.
        val timestamp = sensorTimestampNs / 1e9
        try {
            Protocol.writeImu(
                buffer,
                sendSequence++,
                timestamp,
                orientation,
                gyroscope,
                acceleration,
                accuracy,
                rotationProvider(),
                authTag,
            )
            sock.send(DatagramPacket(buffer.array(), Protocol.IMU_SIZE + authTag.size, destination))
            packetsSent += 1
            val now = SystemClock.elapsedRealtime()
            if (now - lastStatusAt > 1000) {
                lastStatusAt = now
                listener.onImuStatus(
                    "IMU ${sensorHz()} Hz · ${packetsSent} packets" +
                        (if (clockOffsetMs != 0.0) " · clock ${"%.1f".format(clockOffsetMs)} ms" else "")
                )
            }
        } catch (error: Exception) {
            if (!running.get()) return
            lastError = error.message ?: error.javaClass.simpleName
            listener.onImuError("IMU send failed: $lastError")
        }
    }

    /** Copy the gyro and accelerometer values closest to [timestampNs]. */
    private fun pairSensors(timestampNs: Long) {
        if (gyroTimestampNs != 0L && kotlin.math.abs(gyroTimestampNs - timestampNs) < 50_000_000L) {
            System.arraycopy(gyroSample, 0, gyroscope, 0, 3)
        }
        if (accelTimestampNs != 0L && kotlin.math.abs(accelTimestampNs - timestampNs) < 50_000_000L) {
            System.arraycopy(accelSample, 0, acceleration, 0, 3)
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> {
                // Android provides x, y, z, w in that order.
                val orientation = FloatArray(4)
                System.arraycopy(event.values, 0, orientation, 0, minOf(4, event.values.size))
                val timestamp = event.timestamp
                sensorHandler.postDelayed({
                    if (!running.get()) return@postDelayed
                    pairSensors(timestamp)
                    sendPacket(timestamp, orientation)
                }, 2L)
            }
            Sensor.TYPE_GYROSCOPE -> {
                System.arraycopy(event.values, 0, gyroSample, 0, minOf(3, event.values.size))
                gyroTimestampNs = event.timestamp
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                System.arraycopy(event.values, 0, accelSample, 0, minOf(3, event.values.size))
                accelTimestampNs = event.timestamp
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, newAccuracy: Int) {
        if (sensor.type == Sensor.TYPE_ROTATION_VECTOR) {
            accuracy = newAccuracy.coerceIn(0, 3)
        }
    }
}
