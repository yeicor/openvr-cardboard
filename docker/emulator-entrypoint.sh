#!/usr/bin/env bash
# Entry point for the emulator E2E container. The harness boots the emulator,
# runs the desktop app and records both screens; pytest only orchestrates the
# single emulator test and writes JUnit output for CI.
set -euo pipefail

mkdir -p /workspace/artifacts/emulator-e2e
export OVRC_EMULATOR_E2E=1
export PYTHONUNBUFFERED=1

cd /workspace/desktop
exec python3 -m pytest tests/test_emulator_e2e.py \
    -o addopts= -p no:cacheprovider -q -s \
    --junitxml=/workspace/artifacts/emulator-e2e/junit.xml \
    "$@"
