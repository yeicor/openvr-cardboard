"""Monado's Unix socket path limit and the launchers' short-dir fallback."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openvrcardboard.monado_ipc import monado_socket_fits
from openvrcardboard.openxr import MonadoService
from openvrcardboard.openxr_stack import OpenXRStack

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="Monado uses named pipes on Windows, so the path limit does not apply",
)

# A directory component long enough that adding monado_comp_ipc exceeds the
# 108-byte sockaddr_un.sun_path limit.
TOO_LONG = "x" * 120


def test_socket_path_limit_detection(tmp_path):
    assert monado_socket_fits(Path("/run/user/1000"))
    assert monado_socket_fits(tmp_path / "xdg-runtime")
    assert not monado_socket_fits(tmp_path / TOO_LONG)


def test_openxr_stack_uses_a_short_runtime_dir(tmp_path):
    work_dir = tmp_path / TOO_LONG
    stack = OpenXRStack(prefix=tmp_path / "prefix", work_dir=work_dir)
    try:
        env = stack._environment()
        runtime_dir = Path(env["XDG_RUNTIME_DIR"])
        assert monado_socket_fits(runtime_dir)
        assert runtime_dir != work_dir / "xdg-runtime"
        assert runtime_dir.is_dir()
        assert runtime_dir.stat().st_mode & 0o777 == 0o700
        # The service and the OpenXR application must share the directory.
        assert stack._environment()["XDG_RUNTIME_DIR"] == str(runtime_dir)
    finally:
        stack.stop()
    assert not runtime_dir.exists()


def test_monado_service_uses_a_short_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / TOO_LONG))
    service = MonadoService(
        service=tmp_path / "monado-service",
        log_path=tmp_path / "monado-service.log",
    )
    runtime_dir = service._runtime_directory(tmp_path / "cache")
    try:
        assert monado_socket_fits(runtime_dir)
        assert runtime_dir.is_dir()
        assert runtime_dir.stat().st_mode & 0o777 == 0o700
    finally:
        service.stop()
    assert not runtime_dir.exists()


def test_monado_service_keeps_a_fitting_runtime_dir(tmp_path, monkeypatch):
    configured = tmp_path / "runtime"
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(configured))
    service = MonadoService(
        service=tmp_path / "monado-service",
        log_path=tmp_path / "monado-service.log",
    )
    assert service._runtime_directory(tmp_path / "cache") == configured
    service.stop()
