package dev.openvrcardboard.phone

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import android.graphics.Color
import android.hardware.display.DisplayManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.Display
import android.view.Surface
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat

/**
 * Phone-side driver: streams the rotation vector, gyroscope and accelerometer
 * over UDP while showing the desktop's stereo video stream.
 *
 * The activity is landscape-only and immersive. Settings are kept in shared
 * preferences so a phone can be dedicated to the headset without retyping the
 * desktop address on every launch. Tapping the video toggles the settings
 * panel; the hardware back button stops the stream.
 */
class MainActivity : AppCompatActivity(), ImuSender.Listener, VideoClient.Listener {

    companion object {
        private const val TAG = "OpenVRCardboard"
        private const val PREFS = "openvrcardboard"
        private const val KEY_HOST = "host"
        private const val KEY_IMU_PORT = "imu_port"
        private const val KEY_VIDEO_PORT = "video_port"
        private const val KEY_TOKEN = "token"
        private const val KEY_SENSOR_HZ = "sensor_hz"

        private const val DEFAULT_IMU_PORT = 39101
        private const val DEFAULT_VIDEO_PORT = 8765
        private const val DEFAULT_SENSOR_HZ = 120
    }

    private lateinit var hostField: EditText
    private lateinit var imuPortField: EditText
    private lateinit var videoPortField: EditText
    private lateinit var sensorHzField: EditText
    private lateinit var tokenField: EditText
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var statusText: TextView
    private lateinit var panel: LinearLayout
    private lateinit var videoView: StereoView
    private lateinit var videoSurface: VideoSurfaceView
    private lateinit var prefs: SharedPreferences

    private val mainHandler = Handler(Looper.getMainLooper())
    private val frameLock = Any()
    private var pendingFrame: DisplayFrame? = null
    private var framePosted = false

    private var displayManager: DisplayManager? = null
    private var displayRotation = Surface.ROTATION_0
    private var destroyed = false
    private var running = false

    private var imuSender: ImuSender? = null
    private var videoClient: VideoClient? = null
    private var videoFrames = 0L
    private var displayStatsAt = 0L
    private var displayStatsFrames = 0L
    private var hardwareActive = false
    private var formatCodec = "jpeg"
    private var formatWidth = 0
    private var formatHeight = 0
    private var formatLayout = Protocol.LAYOUT_MONO

    private data class DisplayFrame(
        val bitmap: Bitmap,
        val layout: Int,
        val release: () -> Unit,
    )

    private var imuStatus = "IMU idle"
    private var videoStatus = "Video idle"
    private var imuBad = false
    private var videoBad = false

    private val autoHidePanel = Runnable {
        if (running && !imuBad && !videoBad) {
            panel.visibility = View.GONE
        }
    }

    private val frameRunnable = Runnable {
        val frame = synchronized(frameLock) {
            val pending = pendingFrame
            pendingFrame = null
            framePosted = false
            pending
        } ?: return@Runnable
        if (destroyed || !running) {
            frame.release()
            return@Runnable
        }
        videoFrames += 1
        noteDisplayedFrame(frame.bitmap.width, frame.bitmap.height, frame.layout)
        videoView.setFrame(frame.bitmap, frame.layout, frame.release)
    }

    private fun noteDisplayedFrame(width: Int, height: Int, layout: Int) {
        displayStatsFrames += 1
        if (displayStatsAt == 0L) displayStatsAt = SystemClock.elapsedRealtime()
        val now = SystemClock.elapsedRealtime()
        val displayElapsed = now - displayStatsAt
        if (displayElapsed >= 2000L) {
            Log.i(
                TAG,
                "display %.1f fps · %d frames · %dx%d layout=%d".format(
                    displayStatsFrames * 1000.0 / displayElapsed,
                    videoFrames,
                    width,
                    height,
                    layout,
                ),
            )
            displayStatsAt = now
            displayStatsFrames = 0
        }
        if (videoFrames == 1L || videoFrames % 30L == 0L) {
            Log.i(
                TAG,
                "displayed $videoFrames video frames ${width}x$height layout=$layout",
            )
        }
    }

    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) = Unit

        override fun onDisplayRemoved(displayId: Int) = Unit

        override fun onDisplayChanged(displayId: Int) {
            val display = resolveDisplay() ?: return
            if (displayId == display.displayId) {
                displayRotation = display.rotation
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AppContextHolder.install(this)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        bindViews()
        loadPreferences()
        applyIntentOverrides()
        enterImmersiveMode()

        displayManager = getSystemService(DisplayManager::class.java)
        displayRotation = resolveDisplay()?.rotation ?: Surface.ROTATION_0
        applyPreferredRefreshRate()

        startButton.setOnClickListener { startStreaming() }
        stopButton.setOnClickListener { stopStreaming() }
        videoView.setOnClickListener {
            panel.visibility = if (panel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (running) {
                    stopStreaming()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        updateStatus()
    }

    private fun applyIntentOverrides() {
        val intent = intent ?: return
        var changed = false
        if (intent.hasExtra("host")) {
            hostField.setText(intent.getStringExtra("host") ?: "")
            changed = true
        }
        if (intent.hasExtra("imu_port")) {
            imuPortField.setText(intent.getIntExtra("imu_port", DEFAULT_IMU_PORT).toString())
            changed = true
        }
        if (intent.hasExtra("video_port")) {
            videoPortField.setText(intent.getIntExtra("video_port", DEFAULT_VIDEO_PORT).toString())
            changed = true
        }
        if (intent.hasExtra("sensor_hz")) {
            sensorHzField.setText(intent.getIntExtra("sensor_hz", DEFAULT_SENSOR_HZ).toString())
            changed = true
        }
        if (intent.hasExtra("token")) {
            tokenField.setText(intent.getStringExtra("token") ?: "")
            changed = true
        }
        if (changed) {
            val imuPort = imuPortField.text.toString().toIntOrNull() ?: DEFAULT_IMU_PORT
            val videoPort = videoPortField.text.toString().toIntOrNull() ?: DEFAULT_VIDEO_PORT
            val sensorHz = sensorHzField.text.toString().toIntOrNull() ?: DEFAULT_SENSOR_HZ
            savePreferences(
                hostField.text.toString().trim(),
                imuPort,
                videoPort,
                sensorHz,
                tokenField.text.toString().trim(),
            )
        }
        if (intent.getBooleanExtra("autostart", false)) {
            mainHandler.postDelayed({
                if (!destroyed && !isFinishing) startStreaming()
            }, 600)
        }
    }

    private fun bindViews() {
        hostField = findViewById(R.id.hostField)
        imuPortField = findViewById(R.id.imuPortField)
        videoPortField = findViewById(R.id.videoPortField)
        sensorHzField = findViewById(R.id.sensorHzField)
        tokenField = findViewById(R.id.tokenField)
        startButton = findViewById(R.id.startButton)
        stopButton = findViewById(R.id.stopButton)
        statusText = findViewById(R.id.statusText)
        panel = findViewById(R.id.panel)
        videoView = findViewById(R.id.videoView)
        videoSurface = findViewById(R.id.videoSurface)
        videoSurface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                if (running && hardwareActive) {
                    videoClient?.setSurface(holder.surface)
                }
            }

            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int,
            ) = Unit

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                videoClient?.setSurface(null)
            }
        })
    }

    private fun loadPreferences() {
        hostField.setText(prefs.getString(KEY_HOST, ""))
        imuPortField.setText(prefs.getInt(KEY_IMU_PORT, DEFAULT_IMU_PORT).toString())
        videoPortField.setText(prefs.getInt(KEY_VIDEO_PORT, DEFAULT_VIDEO_PORT).toString())
        sensorHzField.setText(prefs.getInt(KEY_SENSOR_HZ, DEFAULT_SENSOR_HZ).toString())
        tokenField.setText(prefs.getString(KEY_TOKEN, ""))
        stopButton.isEnabled = false
    }

    private fun savePreferences(host: String, imuPort: Int, videoPort: Int, sensorHz: Int, token: String) {
        prefs.edit()
            .putString(KEY_HOST, host)
            .putInt(KEY_IMU_PORT, imuPort)
            .putInt(KEY_VIDEO_PORT, videoPort)
            .putInt(KEY_SENSOR_HZ, sensorHz)
            .putString(KEY_TOKEN, token)
            .apply()
    }

    // ------------------------------------------------------------------
    // Streaming lifecycle
    // ------------------------------------------------------------------
    private fun startStreaming() {
        if (running) return
        val host = hostField.text.toString().trim()
        if (host.isEmpty()) {
            toast("Enter the desktop IP address")
            hostField.requestFocus()
            return
        }
        val imuPort = parsePort(imuPortField, DEFAULT_IMU_PORT) ?: return
        val videoPort = parsePort(videoPortField, DEFAULT_VIDEO_PORT) ?: return
        val sensorHz = parseSensorRate() ?: return
        val token = tokenField.text.toString().trim()

        savePreferences(host, imuPort, videoPort, sensorHz, token)
        hideKeyboard()

        running = true
        videoFrames = 0L
        displayStatsAt = 0L
        displayStatsFrames = 0L
        Log.i(TAG, "streaming requested")
        imuBad = false
        videoBad = false
        imuStatus = "IMU starting…"
        videoStatus = "Video connecting…"
        startButton.isEnabled = false
        stopButton.isEnabled = true
        panel.visibility = View.VISIBLE
        updateStatus()

        val sender = ImuSender(host, imuPort, sensorHz, { displayRotation }, this)
        imuSender = sender
        sender.start()

        val metrics = resources.displayMetrics
        val client = VideoClient(
            host,
            videoPort,
            token,
            sensorHz,
            metrics.widthPixels,
            metrics.heightPixels,
            this,
        )
        videoClient = client
        client.start()

        mainHandler.removeCallbacks(autoHidePanel)
        mainHandler.postDelayed(autoHidePanel, 1500)
    }

    private fun stopStreaming(showPanel: Boolean = true) {
        running = false
        mainHandler.removeCallbacks(autoHidePanel)

        val sender = imuSender
        imuSender = null
        sender?.stop()

        val client = videoClient
        videoClient = null
        client?.stop()

        clearPendingFrame()
        client?.setSurface(null)
        hardwareActive = false
        videoView.clear()
        videoSurface.visibility = View.GONE
        videoView.visibility = View.VISIBLE

        imuStatus = "IMU stopped"
        videoStatus = "Video stopped"
        imuBad = false
        videoBad = false
        startButton.isEnabled = true
        stopButton.isEnabled = false
        if (showPanel) panel.visibility = View.VISIBLE
        updateStatus()
    }

    private fun parsePort(field: EditText, fallback: Int): Int? {
        val text = field.text.toString().trim()
        val value = if (text.isEmpty()) fallback else text.toIntOrNull()
        if (value == null || value !in 1..65535) {
            toast("Ports must be between 1 and 65535")
            field.requestFocus()
            return null
        }
        return value
    }

    private fun parseSensorRate(): Int? {
        val text = sensorHzField.text.toString().trim()
        val value = if (text.isEmpty()) DEFAULT_SENSOR_HZ else text.toIntOrNull()
        if (value == null || value !in 10..400) {
            toast("Sensor rate must be between 10 and 400 Hz")
            sensorHzField.requestFocus()
            return null
        }
        return value
    }

    private fun hideKeyboard() {
        val focus = currentFocus ?: return
        val manager = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        manager.hideSoftInputFromWindow(focus.windowToken, 0)
        focus.clearFocus()
    }

    // ------------------------------------------------------------------
    // IMU listener (sensor thread)
    // ------------------------------------------------------------------
    override fun onImuStatus(message: String) {
        mainHandler.post {
            if (destroyed) return@post
            imuBad = false
            imuStatus = message
            Log.i(TAG, "imu: $message")
            updateStatus()
        }
    }

    override fun onImuError(message: String) {
        mainHandler.post {
            if (destroyed) return@post
            imuBad = true
            imuStatus = message
            panel.visibility = View.VISIBLE
            updateStatus()
        }
    }

    // ------------------------------------------------------------------
    // Video listener (OkHttp reader thread)
    // ------------------------------------------------------------------
    override fun onVideoStatus(message: String) {
        mainHandler.post {
            if (destroyed) return@post
            videoBad = false
            videoStatus = message
            Log.i(TAG, "video: $message")
            updateStatus()
        }
    }

    override fun onVideoError(message: String) {
        mainHandler.post {
            if (destroyed) return@post
            videoBad = true
            videoStatus = message
            Log.w(TAG, "video error: $message")
            panel.visibility = View.VISIBLE
            updateStatus()
        }
    }

    override fun onVideoFrame(bitmap: Bitmap, layout: Int) {
        val client = videoClient
        if (client == null || destroyed || !running) {
            bitmap.recycle()
            return
        }
        if (hardwareActive) {
            // A late JPEG frame after a fallback: show it and leave the
            // surface path; the next format message normally does this too.
            hardwareActive = false
            videoSurface.visibility = View.GONE
            videoView.visibility = View.VISIBLE
            client.setSurface(null)
        }
        val frame = DisplayFrame(bitmap, layout) { client.release(bitmap) }
        val replaced = synchronized(frameLock) {
            val previous = pendingFrame
            pendingFrame = frame
            if (!framePosted) {
                framePosted = true
                mainHandler.post(frameRunnable)
            }
            previous
        }
        // Only the newest decoded frame is shown; release a frame that was
        // superseded before the UI thread could draw it back to the pool.
        replaced?.release?.invoke()
    }

    override fun onVideoFormat(format: VideoFormat) {
        mainHandler.post {
            if (destroyed || !running) return@post
            formatCodec = format.codec
            formatWidth = format.width
            formatHeight = format.height
            formatLayout = format.layout
            val client = videoClient
            if (format.codec == "jpeg") {
                hardwareActive = false
                client?.setSurface(null)
                videoSurface.visibility = View.GONE
                videoView.visibility = View.VISIBLE
            } else {
                hardwareActive = true
                videoView.visibility = View.GONE
                videoSurface.setVideoSize(format.width, format.height)
                videoSurface.visibility = View.VISIBLE
                val surface = videoSurface.holder.surface
                if (surface != null && surface.isValid) {
                    client?.setSurface(surface)
                }
                videoStatus = "Video ${format.codec.uppercase()} " +
                    "${format.width}x${format.height} @ ${format.fps} fps"
                updateStatus()
            }
            Log.i(
                TAG,
                "video format ${format.codec} ${format.width}x${format.height} " +
                    "layout=${format.layout} fps=${format.fps}",
            )
        }
    }

    override fun onHardwareFrame() {
        mainHandler.post {
            if (destroyed || !running || !hardwareActive) return@post
            videoFrames += 1
            noteDisplayedFrame(formatWidth, formatHeight, formatLayout)
        }
    }

    override fun onVideoRecenter() {
        mainHandler.post {
            if (!destroyed) toast("Desktop recentered")
        }
    }

    private fun clearPendingFrame() {
        val pending = synchronized(frameLock) {
            val current = pendingFrame
            pendingFrame = null
            framePosted = false
            current
        }
        mainHandler.removeCallbacks(frameRunnable)
        pending?.release?.invoke()
    }

    // ------------------------------------------------------------------
    // Display / window
    // ------------------------------------------------------------------
    @Suppress("DEPRECATION")
    private fun resolveDisplay(): Display? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display
        } else {
            windowManager.defaultDisplay
        }
    }

    /**
     * Ask the panel for its highest refresh mode. Headset streaming only feels
     * right at 90 Hz or more, and many phones keep 60 Hz unless an app requests
     * a high-refresh mode explicitly.
     */
    @Suppress("DEPRECATION")
    private fun applyPreferredRefreshRate() {
        val display = resolveDisplay() ?: return
        val modes = try {
            display.supportedModes
        } catch (_: Exception) {
            null
        }
        if (modes.isNullOrEmpty()) return
        val best = modes
            .filter { it.refreshRate >= 89.0f }
            .maxWithOrNull(
                compareBy<Display.Mode>(
                    { it.refreshRate },
                    { it.physicalWidth * it.physicalHeight },
                )
            )
            ?: modes.maxByOrNull { it.refreshRate }
            ?: return
        val params = window.attributes
        if (params.preferredDisplayModeId != best.modeId) {
            params.preferredDisplayModeId = best.modeId
            window.attributes = params
        }
        val current = try {
            display.refreshRate
        } catch (_: Exception) {
            best.refreshRate
        }
        Log.i(
            TAG,
            "display mode ${best.modeId}: ${best.physicalWidth}x${best.physicalHeight} " +
                "@ %.0f Hz (panel now %.0f Hz)".format(best.refreshRate, current),
        )
    }

    private fun enterImmersiveMode() {
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) enterImmersiveMode()
    }

    override fun onResume() {
        super.onResume()
        displayManager?.registerDisplayListener(displayListener, mainHandler)
        displayRotation = resolveDisplay()?.rotation ?: displayRotation
        if (running) applyPreferredRefreshRate()
    }

    override fun onPause() {
        displayManager?.unregisterDisplayListener(displayListener)
        super.onPause()
    }

    override fun onDestroy() {
        destroyed = true
        stopStreaming(showPanel = false)
        mainHandler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    private fun updateStatus() {
        statusText.text = "$imuStatus\n$videoStatus"
        statusText.setTextColor(
            if (imuBad || videoBad) Color.rgb(255, 123, 114) else Color.rgb(159, 227, 165)
        )
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
