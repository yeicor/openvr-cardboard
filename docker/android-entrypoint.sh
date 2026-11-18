#!/usr/bin/env bash
# Entry point for the Android container: run the JVM unit tests and build the
# debug APK. Extra arguments replace the default Gradle task list.
set -euo pipefail

cd /workspace/android

if [ "$#" -gt 0 ]; then
    exec gradle --no-daemon "$@"
fi

exec gradle --no-daemon testDebugUnitTest assembleDebug
