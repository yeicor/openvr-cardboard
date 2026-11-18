package dev.openvrcardboard.phone

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Draws the complete stereo JPEG across the panel. Side-by-side and top-bottom
 * frames are already a complete stereo pair, so no per-eye logic is needed on
 * the phone; a cardboard-style viewer sends each half to one eye.
 *
 * Bitmaps are owned by [VideoClient] and returned through the release callback
 * once they are no longer on screen, which lets the decoder reuse them through
 * Android's `inBitmap` path instead of allocating a new frame every time.
 */
class StereoView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)
    private var frame: Bitmap? = null
    private var frameRelease: (() -> Unit)? = null
    private var layout = Protocol.LAYOUT_MONO
    private val destination = RectF()

    fun setFrame(bitmap: Bitmap, layout: Int, release: (() -> Unit)? = null) {
        val previous = frameRelease
        frame = bitmap
        frameRelease = release
        this.layout = layout
        invalidate()
        previous?.invoke()
    }

    fun clear() {
        val previous = frameRelease
        frame = null
        frameRelease = null
        invalidate()
        previous?.invoke()
    }

    fun currentLayout(): Int = layout

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)
        val bitmap = frame ?: return
        if (bitmap.isRecycled || width == 0 || height == 0) return
        val scale = min(
            width.toFloat() / bitmap.width.toFloat(),
            height.toFloat() / bitmap.height.toFloat(),
        )
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val left = (width - drawWidth) / 2f
        val top = (height - drawHeight) / 2f
        destination.set(left, top, left + drawWidth, top + drawHeight)
        canvas.drawBitmap(bitmap, null, destination, paint)
    }

    override fun onDetachedFromWindow() {
        clear()
        super.onDetachedFromWindow()
    }
}
