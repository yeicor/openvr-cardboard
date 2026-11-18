package dev.openvrcardboard.phone

/** Small application-scoped holder so sensor helpers can reach a context. */
object AppContextHolder {
    lateinit var context: android.content.Context
        private set

    fun install(context: android.content.Context) {
        this.context = context.applicationContext
    }
}
