
import pytest

from openvrcardboard import gpu
from openvrcardboard.config import PhoneConfig
from openvrcardboard.encoder import (
    FfmpegVideoEncoder,
    IvfDemuxer,
    NvencVideoEncoder,
    TsDemuxer,
    _is_keyframe,
    choose_stream_codec,
    parameter_sets,
    select_stream_encoder,
)
from openvrcardboard.gpu import GpuCapabilities


def _ts_packet(pid: int, payload: bytes, start: bool, counter: int) -> bytes:
    header = bytes(
        (
            0x47,
            (0x40 if start else 0x00) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0x10 | (counter & 0x0F),
        )
    )
    return (header + payload)[:188].ljust(188, b"\xFF")


def _section(table_id: int, body: bytes) -> bytes:
    length = len(body) + 4
    return bytes((table_id, 0xB0 | ((length >> 8) & 0x0F), length & 0xFF)) + body


def _crc32(data: bytes) -> bytes:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xEDB88320 if crc & 1 else crc >> 1
    return (crc ^ 0xFFFFFFFF).to_bytes(4, "big")


def _pat(pmt_pid: int = 0x1000) -> bytes:
    body = (1).to_bytes(2, "big") + b"\xC1\x00\x00" + (1).to_bytes(2, "big") + (
        0xE000 | pmt_pid
    ).to_bytes(2, "big")
    return _section(0x00, body + _crc32(b"\x00" + body))


def _pmt(video_pid: int = 0x0101) -> bytes:
    body = (
        (1).to_bytes(2, "big")
        + b"\xC1\x00\x00"
        + (0xE000 | video_pid).to_bytes(2, "big")
        + b"\xF0\x00"
        + bytes((0x24,))
        + (0xE000 | video_pid).to_bytes(2, "big")
        + b"\xF0\x00"
    )
    return _section(0x02, body + _crc32(b"\x02" + body))


def _pes(keyframe: bool, payload: bytes) -> bytes:
    return (
        b"\x00\x00\x01\xE0\x00\x00\x80"
        + bytes((0x40 if keyframe else 0x00, 0x00))
        + payload
    )


def test_ts_demuxer_returns_annex_b_units_and_keyframes():
    demuxer = TsDemuxer()
    pat = _ts_packet(0, b"\x00" + _pat(), True, 0)
    pmt = _ts_packet(0x1000, b"\x00" + _pmt(), True, 0)
    keyframe = b"\x00\x00\x00\x01\x26\x01\xAA"
    delta = b"\x00\x00\x00\x01\x02\x01\xBB"
    first = _ts_packet(0x0101, _pes(True, keyframe), True, 0)
    second = _ts_packet(0x0101, _pes(False, delta), True, 1)
    units = demuxer.feed(pat + pmt + first + second)
    assert len(units) == 1
    data, key = units[0]
    assert key is True
    assert data.startswith(b"\x00\x00\x00\x01\x26")
    assert demuxer.flush()[0][1] is False


def test_ivf_demuxer_handles_partial_frames():
    header = b"DKIF" + (0).to_bytes(2, "little") + (32).to_bytes(2, "little")
    header += b"AV01" + bytes(20)
    frames = [b"\x0A" * 5, b"\x0B" * 9]
    payload = header
    for frame in frames:
        payload += len(frame).to_bytes(4, "little") + b"\x00" * 8 + frame
    demuxer = IvfDemuxer()
    units = []
    for index in range(0, len(payload), 7):
        units.extend(demuxer.feed(payload[index : index + 7]))
    assert units == frames


def test_parameter_sets_extracts_vps_sps_pps():
    hevc = (
        b"\x00\x00\x00\x01\x40\x01VPS"
        b"\x00\x00\x00\x01\x42\x01SPS"
        b"\x00\x00\x00\x01\x44\x01PPS"
        b"\x00\x00\x00\x01\x26\x01IDR"
    )
    result = parameter_sets("hevc", hevc)
    assert b"VPS" in result and b"SPS" in result and b"PPS" in result
    assert b"IDR" not in result
    assert _is_keyframe("hevc", hevc)

    h264 = (
        b"\x00\x00\x01\x67SPS"
        b"\x00\x00\x01\x68PPS"
        b"\x00\x00\x01\x65IDR"
    )
    result = parameter_sets("h264", h264)
    assert b"SPS" in result and b"PPS" in result and b"IDR" not in result
    assert _is_keyframe("h264", h264)


def test_choose_stream_codec_respects_configuration(monkeypatch):
    caps = GpuCapabilities(
        available=True,
        name="Test GPU",
        ffmpeg="/usr/bin/ffmpeg",
        encoders={"hevc_nvenc", "av1_nvenc", "libx264"},
    )
    monkeypatch.setattr(
        "openvrcardboard.encoder.detect_gpu", lambda: caps
    )
    assert choose_stream_codec(PhoneConfig(codec="auto"))[0] == "hevc"
    assert choose_stream_codec(PhoneConfig(codec="av1"))[0] == "av1"
    assert choose_stream_codec(PhoneConfig(codec="h264"))[0] == "h264"
    assert choose_stream_codec(PhoneConfig(codec="jpeg"))[0] == "jpeg"

    unavailable = GpuCapabilities(
        available=False,
        ffmpeg="/usr/bin/ffmpeg",
        encoders={"libx264"},
        reason="no GPU",
    )
    monkeypatch.setattr(
        "openvrcardboard.encoder.detect_gpu", lambda: unavailable
    )
    codec, reason = choose_stream_codec(PhoneConfig(codec="auto"))
    assert codec == "jpeg"
    assert "no GPU" in reason
    # Auto still prefers JPEG, but an explicit H.264 request can use the CPU.
    assert choose_stream_codec(PhoneConfig(codec="h264"))[0] == "h264"
    assert choose_stream_codec(PhoneConfig(codec="h264", encoder="nvenc"))[0] == "jpeg"


def test_x264_selection_is_explicit_and_h264_only(monkeypatch):
    caps = GpuCapabilities(
        available=False,
        ffmpeg="/usr/bin/ffmpeg",
        encoders={"libx264"},
        reason="no NVIDIA GPU",
    )
    monkeypatch.setattr("openvrcardboard.encoder.detect_gpu", lambda: caps)
    selection = select_stream_encoder(PhoneConfig(codec="auto", encoder="x264"))
    assert selection.codec == "h264"
    assert selection.encoder == "libx264"
    assert "CPU" in selection.reason
    assert select_stream_encoder(PhoneConfig(codec="h264", encoder="cpu")).codec == "h264"
    assert select_stream_encoder(PhoneConfig(codec="hevc", encoder="x264")).codec == "jpeg"

    missing = GpuCapabilities(available=False, ffmpeg="/usr/bin/ffmpeg", reason="no encoder")
    monkeypatch.setattr("openvrcardboard.encoder.detect_gpu", lambda: missing)
    selection = select_stream_encoder(PhoneConfig(codec="h264", encoder="x264"))
    assert selection.codec == "jpeg"
    assert "libx264" in selection.reason


def test_cpu_encoder_command_uses_libx264_realtime_options():
    encoder = FfmpegVideoEncoder(
        "h264",
        640,
        360,
        30,
        bitrate_mbps=8,
        ffmpeg="/bin/true",
        encoder="libx264",
    )
    command = encoder._command()
    assert "libx264" in command
    assert "ultrafast" in command
    assert "zerolatency" in command
    assert "-rc" not in command
    assert "-x264-params" in command
    assert any("repeat-headers=1" in argument for argument in command)


def test_cpu_encoder_produces_annex_b_h264_access_units():
    ffmpeg = gpu.find_ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")
    if "libx264" not in gpu._query_encoders(ffmpeg)[0]:
        pytest.skip("ffmpeg has no libx264 encoder")
    import time as _time

    import numpy as np

    encoder = FfmpegVideoEncoder(
        "h264", 320, 180, 15, bitrate_mbps=2, ffmpeg=ffmpeg, encoder="libx264"
    )
    packets = []
    try:
        encoder.wait_ready()
        for index in range(40):
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image[:, :] = (index * 5) % 256
            image[:, : 160] = (20, 40, 180)
            packets.extend(encoder.encode(image, timeout=0.2))
            _time.sleep(0.01)
        deadline = _time.monotonic() + 5.0
        while not packets and _time.monotonic() < deadline:
            packets.extend(encoder.encode(image, timeout=0.2))
    finally:
        encoder.close()
    assert packets, encoder.error or "no H.264 packets were produced"
    assert any(packet.keyframe for packet in packets)
    keyframe = next(packet for packet in packets if packet.keyframe)
    assert b"\x00\x00\x00\x01" in keyframe.data or b"\x00\x00\x01" in keyframe.data
    assert _is_keyframe("h264", keyframe.data)
    assert parameter_sets("h264", packets[-1].data) or any(
        parameter_sets("h264", packet.data) for packet in packets
    )


def test_gpu_detection_uses_nvenc_capabilities(monkeypatch):
    monkeypatch.delenv("OPENVRCARDBOARD_DISABLE_GPU", raising=False)
    monkeypatch.setattr(gpu, "find_ffmpeg", lambda configured="": "/fake/ffmpeg")
    monkeypatch.setattr(
        gpu, "_query_nvidia", lambda: ("RTX Test", "999.1", 16384, "")
    )
    monkeypatch.setattr(
        gpu,
        "_query_encoders",
        lambda path: ({"hevc_nvenc", "av1_nvenc", "h264_nvenc"}, ""),
    )
    gpu.reset_cache()
    try:
        caps = gpu.detect_gpu(refresh=True)
        assert caps.available
        assert caps.name == "RTX Test"
        assert caps.memory_mb == 16384
        assert caps.supports("hevc")
        assert caps.best_codec() == "hevc"
        assert "NVENC" in caps.summary()
    finally:
        gpu.reset_cache()


def test_gpu_detection_can_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("OPENVRCARDBOARD_DISABLE_GPU", "1")
    caps = gpu.detect_gpu()
    assert not caps.available
    assert "disabled" in caps.reason.lower()


def test_bundled_ffmpeg_uses_imageio_fallback(monkeypatch, tmp_path):
    import sys
    import types

    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    module = types.SimpleNamespace(get_ffmpeg_exe=lambda: str(fake))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", module)
    monkeypatch.delenv("OPENVRCARDBOARD_FFMPEG", raising=False)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "_MEIPASS", "", raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "app"))
    assert gpu.find_ffmpeg() == str(fake)


def test_find_ffmpeg_prefers_an_explicit_path(monkeypatch, tmp_path):
    import sys

    fake = tmp_path / "custom-ffmpeg"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    assert gpu.find_ffmpeg(str(fake)) == str(fake)
    assert gpu.find_ffmpeg(str(tmp_path / "missing")) == "/usr/bin/ffmpeg"


def test_nvenc_encoder_rejects_unknown_codec():
    with pytest.raises(Exception):
        NvencVideoEncoder("vp9", 640, 360, 30)
