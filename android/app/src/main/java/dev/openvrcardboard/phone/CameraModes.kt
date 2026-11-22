package dev.openvrcardboard.phone

/**
 * Shared camera resolution/FPS fallback ladder.
 *
 * Keep the default entries in sync with `protocol/camera_modes.txt` and the
 * desktop `openvrcardboard.camera_modes` module. A camera first tries the
 * requested target, then resumes at the canonical entry closest to that
 * target. Capture implementations add their own final "whatever the camera
 * accepts" mode.
 */
internal data class CameraModeRequest(
    val width: Int,
    val height: Int,
    val fps: Int,
)

internal object CameraModes {
    val DEFAULT_MODES: List<CameraModeRequest> = listOf(
        CameraModeRequest(1920, 1080, 120),
        CameraModeRequest(1280, 720, 120),
        CameraModeRequest(1920, 1080, 90),
        CameraModeRequest(1280, 720, 90),
        CameraModeRequest(1920, 1080, 60),
        CameraModeRequest(1280, 720, 60),
        CameraModeRequest(1920, 1080, 30),
        CameraModeRequest(1280, 720, 30),
    )
    val DEFAULT_MODE: CameraModeRequest = DEFAULT_MODES.first()

    /**
     * Return the target followed by the canonical fallback modes.
     *
     * The fallback starts at the ladder entry closest to the target among
     * modes that do not exceed its resolution or frame rate. That keeps a
     * manual 720p60 request from retrying 1080p120 while still giving the
     * requested target one chance before falling back.
     */
    fun attempts(target: CameraModeRequest?): List<CameraModeRequest> {
        if (target == null ||
            target.width <= 0 ||
            target.height <= 0 ||
            target.fps <= 0
        ) {
            return DEFAULT_MODES
        }

        val attempts = mutableListOf(target)
        val eligible = DEFAULT_MODES.filter {
            it.width <= target.width &&
                it.height <= target.height &&
                it.fps <= target.fps
        }
        if (eligible.isEmpty()) return attempts

        val closest = eligible.minByOrNull { distance(it, target) } ?: return attempts
        val start = eligible.indexOf(closest)
        for (mode in eligible.drop(start)) {
            if (mode != target && mode !in attempts) attempts.add(mode)
        }
        return attempts
    }

    private fun distance(mode: CameraModeRequest, target: CameraModeRequest): Double {
        return kotlin.math.abs(mode.width - target.width).toDouble() /
            maxOf(target.width, 1) +
            kotlin.math.abs(mode.height - target.height).toDouble() /
            maxOf(target.height, 1) +
            kotlin.math.abs(mode.fps - target.fps).toDouble() /
            maxOf(target.fps, 1)
    }
}
