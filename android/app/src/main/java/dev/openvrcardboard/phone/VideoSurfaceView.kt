package dev.openvrcardboard.phone

import android.content.Context
import android.util.AttributeSet
import android.view.SurfaceView
import kotlin.math.roundToInt

/**
 * A SurfaceView that keeps the decoded stereo frame's aspect ratio.
 *
 * MediaCodec renders directly into this surface; the compositor scales it to
 * the panel. The view only has to avoid stretching a 2400x1080 stream onto a
 * 16:9 phone screen, which would displace the two eye images relative to the
 * physical lenses.
 */
class VideoSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : SurfaceView(context, attrs) {

    private var aspect = 0.0

    fun setVideoSize(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        val value = width.toDouble() / height.toDouble()
        if (kotlin.math.abs(value - aspect) < 1e-6) return
        aspect = value
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = MeasureSpec.getSize(heightMeasureSpec)
        if (aspect <= 0.0 || width == 0 || height == 0) {
            setMeasuredDimension(width, height)
            return
        }
        val panelAspect = width.toDouble() / height.toDouble()
        if (panelAspect > aspect) {
            // Panel is wider than the video: fill the height.
            val targetWidth = (height * aspect).roundToInt().coerceAtMost(width)
            setMeasuredDimension(targetWidth, height)
        } else {
            val targetHeight = (width / aspect).roundToInt().coerceAtMost(height)
            setMeasuredDimension(width, targetHeight)
        }
    }
}
