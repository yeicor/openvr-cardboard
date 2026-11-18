"""Low-latency HEVC/AV1/H.264 streaming encoder backed by ffmpeg.

The desktop already has ffmpeg on the systems this app targets. Instead of
adding a Python codec dependency, the encoder runs ``ffmpeg`` as a child
process, feeds it raw side-by-side stereo frames and demuxes the MPEG-TS
output back into access units that Android's ``MediaCodec`` can consume.

NVIDIA NVENC is used when a GPU and its encoders are available. The same
pipeline can run ``libx264`` on the CPU, which keeps the hardware decode path
exercisable in CI, on non-NVIDIA machines, and in the bundled ffmpeg.

Only the video stream crosses the process boundary; there is no temporary
file and no per-frame process spawn. The TS demuxer is deliberately small but
understands PAT/PMT, PES continuation packets and the PES random-access flag,
so each returned packet is a complete Annex-B access unit with a keyframe bit.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import PhoneConfig
from .gpu import GpuCapabilities, detect_gpu

LOGGER = logging.getLogger(__name__)

CODEC_ENCODERS = {
    "hevc": "hevc_nvenc",
    "h264": "h264_nvenc",
    "av1": "av1_nvenc",
}
CPU_ENCODER = "libx264"
CODEC_MIME = {
    "hevc": "video/hevc",
    "h264": "video/avc",
    "av1": "video/av01",
}
CODEC_WIRE = {"jpeg": 0, "hevc": 1, "av1": 2, "h264": 3}


class EncoderError(RuntimeError):
    """Raised when a hardware encoder cannot start or fails mid-stream."""


@dataclass(slots=True)
class EncodedPacket:
    data: bytes
    codec: str
    keyframe: bool
    width: int
    height: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class StreamSelection:
    """Chosen wire codec and ffmpeg encoder for the phone stream."""

    codec: str
    encoder: str
    reason: str


def codec_wire_value(codec: str) -> int:
    return CODEC_WIRE.get((codec or "jpeg").lower(), 0)


def codec_mime(codec: str) -> str:
    return CODEC_MIME.get((codec or "").lower(), "")


class TsDemuxer:
    """Extract video access units and keyframe flags from MPEG-TS bytes."""

    PACKET = 188

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pmt_pid: Optional[int] = None
        self._video_pid: Optional[int] = None
        self._unit = bytearray()
        self._unit_keyframe = False
        self._unit_started = False

    def feed(self, data: bytes) -> list[tuple[bytes, bool]]:
        if data:
            self._buffer.extend(data)
        units: list[tuple[bytes, bool]] = []
        while len(self._buffer) >= self.PACKET:
            packet = bytes(self._buffer[: self.PACKET])
            del self._buffer[: self.PACKET]
            if packet[0] != 0x47:
                # Resynchronise after a lost byte rather than poisoning the
                # whole stream.
                self._buffer.extend(packet[1:])
                continue
            unit = self._packet(packet)
            if unit is not None:
                units.append(unit)
        return units

    def flush(self) -> list[tuple[bytes, bool]]:
        units = self.feed(b"")
        if self._unit_started and self._unit:
            units.append((bytes(self._unit), self._unit_keyframe))
        self._unit = bytearray()
        self._unit_started = False
        return units

    def _packet(self, packet: bytes) -> Optional[tuple[bytes, bool]]:
        pid = ((packet[1] & 0x1F) << 8) | packet[2]
        pusi = bool(packet[1] & 0x40)
        offset = 4
        adaptation = (packet[3] >> 4) & 0x03
        if adaptation & 0x02:
            if offset >= self.PACKET:
                return None
            offset += 1 + packet[4]
        if not adaptation & 0x01 or offset >= self.PACKET:
            return None
        payload = packet[offset:]

        if pid == 0:
            self._parse_pat(payload, pusi)
            return None
        if self._pmt_pid is not None and pid == self._pmt_pid:
            self._parse_pmt(payload, pusi)
            return None
        if self._video_pid is None or pid != self._video_pid:
            return None
        return self._video_payload(payload, pusi)

    def _section(self, payload: bytes, pusi: bool) -> Optional[bytes]:
        if pusi:
            if not payload:
                return None
            pointer = payload[0]
            section = payload[1 + pointer :]
        else:
            section = payload
        if len(section) < 3:
            return None
        length = ((section[1] & 0x0F) << 8) | section[2]
        end = 3 + length - 4  # exclude CRC32
        if len(section) < end:
            # A section spanning several TS packets is rare for PAT/PMT at the
            # sizes used here; ignore the fragment rather than guessing.
            return None
        return section[:end]

    def _parse_pat(self, payload: bytes, pusi: bool) -> None:
        section = self._section(payload, pusi)
        if section is None or len(section) < 12:
            return
        if section[0] != 0x00:
            return
        for index in range(8, len(section) - 3, 4):
            program = (section[index] << 8) | section[index + 1]
            pid = ((section[index + 2] & 0x1F) << 8) | section[index + 3]
            if program:
                self._pmt_pid = pid
                return

    def _parse_pmt(self, payload: bytes, pusi: bool) -> None:
        section = self._section(payload, pusi)
        if section is None or len(section) < 16:
            return
        if section[0] != 0x02:
            return
        program_info = ((section[10] & 0x0F) << 8) | section[11]
        index = 12 + program_info
        while index + 5 <= len(section):
            stream_type = section[index]
            pid = ((section[index + 1] & 0x1F) << 8) | section[index + 2]
            info_length = ((section[index + 3] & 0x0F) << 8) | section[index + 4]
            if stream_type in (0x24, 0x1B, 0x0F, 0x02):
                self._video_pid = pid
                return
            index += 5 + info_length

    def _video_payload(self, payload: bytes, pusi: bool) -> Optional[tuple[bytes, bool]]:
        if pusi:
            finished = None
            if self._unit_started and self._unit:
                finished = (bytes(self._unit), self._unit_keyframe)
            self._unit = bytearray()
            self._unit_started = True
            self._unit_keyframe = False
            if len(payload) < 9 or payload[:3] != b"\x00\x00\x01":
                # Not a PES start; keep it as the beginning of an access unit.
                self._unit.extend(payload)
                return finished
            flags2 = payload[7]
            self._unit_keyframe = bool(flags2 & 0x40)
            header_length = payload[8]
            start = 9 + header_length
            if start <= len(payload):
                self._unit.extend(payload[start:])
            return finished
        if self._unit_started:
            self._unit.extend(payload)
        return None


def _is_keyframe(codec: str, data: bytes) -> bool:
    """Detect an IRAP/IDR access unit from Annex-B NAL units."""
    if not data:
        return False
    if codec == "hevc":
        index = data.find(b"\x00\x00\x01")
        while index >= 0:
            nal = index + 3
            if nal + 1 < len(data):
                nal_type = (data[nal] >> 1) & 0x3F
                if 16 <= nal_type <= 21:
                    return True
            index = data.find(b"\x00\x00\x01", nal)
        return False
    if codec == "h264":
        index = data.find(b"\x00\x00\x01")
        while index >= 0:
            nal = index + 3
            if nal < len(data):
                nal_type = data[nal] & 0x1F
                if nal_type in (5, 7):
                    return True
            index = data.find(b"\x00\x00\x01", nal)
        return False
    return False


def parameter_sets(codec: str, data: bytes) -> bytes:
    """Return the Annex-B parameter sets from a keyframe, if present."""
    if not data:
        return b""
    if codec == "hevc":
        wanted = {32, 33, 34}  # VPS, SPS, PPS
    elif codec == "h264":
        wanted = {7, 8}  # SPS, PPS
    else:
        return b""
    result = bytearray()
    index = data.find(b"\x00\x00\x01")
    while index >= 0:
        start = index
        nal = index + 3
        next_index = data.find(b"\x00\x00\x01", nal)
        end = len(data) if next_index < 0 else next_index
        if nal < len(data):
            if codec == "hevc":
                nal_type = (data[nal] >> 1) & 0x3F
            else:
                nal_type = data[nal] & 0x1F
            if nal_type in wanted:
                result.extend(data[start:end])
        index = next_index
    return bytes(result)


class IvfDemuxer:
    """Split IVF output into individual AV1 temporal units."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._header_done = False

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data or b"")
        units: list[bytes] = []
        if not self._header_done:
            if len(self._buffer) < 32:
                return units
            if self._buffer[:4] != b"DKIF":
                raise EncoderError("ffmpeg produced invalid IVF output")
            header_length = int.from_bytes(self._buffer[6:8], "little")
            skip = max(32, header_length)
            if len(self._buffer) < skip:
                return units
            del self._buffer[:skip]
            self._header_done = True
        while len(self._buffer) >= 12:
            size = int.from_bytes(self._buffer[:4], "little")
            if size <= 0 or size > 32 * 1024 * 1024:
                raise EncoderError(f"invalid IVF frame size {size}")
            end = 12 + size
            if len(self._buffer) < end:
                break
            units.append(bytes(self._buffer[12:end]))
            del self._buffer[:end]
        return units


class FfmpegVideoEncoder:
    """One ffmpeg process encoding a side-by-side stereo stream."""

    def __init__(
        self,
        codec: str,
        width: int,
        height: int,
        fps: int,
        bitrate_mbps: float = 40.0,
        keyframe_interval_s: float = 1.0,
        ffmpeg: str = "",
        preset: str = "p1",
        encoder: str = "",
    ):
        codec = (codec or "").lower()
        if codec not in CODEC_ENCODERS:
            raise EncoderError(f"unsupported stream codec {codec!r}")
        self.codec = codec
        self.encoder = encoder or CODEC_ENCODERS[codec]
        if self.encoder not in set(CODEC_ENCODERS.values()) | {CPU_ENCODER}:
            raise EncoderError(f"unsupported video encoder {self.encoder!r}")
        if self.encoder == CPU_ENCODER and codec != "h264":
            raise EncoderError("libx264 only encodes H.264")
        self.container = "ivf" if codec == "av1" else "mpegts"
        self.width = max(2, int(width) & ~1)
        self.height = max(2, int(height) & ~1)
        self.fps = max(1, int(fps))
        self.bitrate_mbps = max(1.0, float(bitrate_mbps))
        self.keyframe_interval = max(
            1, int(round(self.fps * max(0.1, float(keyframe_interval_s))))
        )
        if not ffmpeg:
            from .gpu import find_ffmpeg

            ffmpeg = find_ffmpeg()
        self.ffmpeg = ffmpeg
        if not self.ffmpeg:
            raise EncoderError("ffmpeg is not installed or not bundled")
        self._preset = preset
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._process: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stderr = ""
        self._error = ""
        self._packets = 0
        self._bytes = 0
        self._use_i420 = False
        try:
            import cv2  # noqa: F401
        except ImportError:
            self._use_i420 = False
        else:
            self._use_i420 = True

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def error(self) -> str:
        return self._error

    @property
    def stats(self) -> tuple[int, int]:
        return self._packets, self._bytes

    def _command(self) -> list[str]:
        encoder = self.encoder
        bitrate = int(self.bitrate_mbps * 1_000_000)
        pixel_format = "yuv420p" if self._use_i420 else "rgb24"
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pixel_format,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        if encoder == CPU_ENCODER:
            # libx264's realtime mode: no B-frames, no lookahead, and SPS/PPS
            # repeated in-band before every IDR so MediaCodec can start or
            # recover without an out-of-band CSD update.
            command.extend(
                [
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-x264-params",
                    "repeat-headers=1:scenecut=0",
                    "-b:v",
                    str(bitrate),
                    "-maxrate",
                    str(bitrate),
                    "-bufsize",
                    str(max(bitrate // 2, 1_000_000)),
                ]
            )
        else:
            command.extend(
                [
                    "-preset",
                    self._preset,
                    "-tune",
                    "ull",
                    "-rc",
                    "cbr",
                    "-b:v",
                    str(bitrate),
                    "-maxrate",
                    str(bitrate),
                    "-bufsize",
                    str(max(bitrate // 2, 1_000_000)),
                ]
            )
        command.extend(
            [
                "-g",
                str(self.keyframe_interval),
                "-keyint_min",
                str(self.keyframe_interval),
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-f",
                self.container,
                "-flush_packets",
                "1",
                "pipe:1",
            ]
        )
        return command

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise EncoderError(f"could not start ffmpeg: {exc}") from exc
        self._reader = threading.Thread(
            target=self._read_loop, name=f"openvrcardboard-{self.codec}-encoder", daemon=True
        )
        self._reader.start()

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        demuxer = IvfDemuxer() if self.container == "ivf" else TsDemuxer()
        index = 0
        try:
            while True:
                chunk = process.stdout.read(188 * 8)
                if not chunk:
                    break
                if self.container == "ivf":
                    units = [(unit, False) for unit in demuxer.feed(chunk)]
                else:
                    units = demuxer.feed(chunk)
                for unit, keyframe in units:
                    if not keyframe:
                        keyframe = _is_keyframe(self.codec, unit)
                    if not keyframe and self.codec == "av1":
                        keyframe = index % self.keyframe_interval == 0
                    index += 1
                    self._enqueue(unit, keyframe)
            if self.container == "ivf":
                tail = [(unit, False) for unit in demuxer.feed(b"")]
            else:
                tail = demuxer.flush()
            for unit, keyframe in tail:
                if not keyframe:
                    keyframe = _is_keyframe(self.codec, unit)
                if not keyframe and self.codec == "av1":
                    keyframe = index % self.keyframe_interval == 0
                index += 1
                self._enqueue(unit, keyframe)
        except Exception as exc:  # pragma: no cover - reader thread guard
            self._error = str(exc)
        finally:
            self._stderr = self._read_stderr(process)

    def _enqueue(self, unit: bytes, keyframe: bool) -> None:
        packet = EncodedPacket(
            data=unit,
            codec=self.codec,
            keyframe=keyframe,
            width=self.width,
            height=self.height,
            timestamp=time.monotonic(),
        )
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            # Drop the oldest packet, never stall the encoder.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                pass

    @staticmethod
    def _read_stderr(process: subprocess.Popen) -> str:
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _prepare(self, image) -> bytes:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            elif array.shape[2] == 4:
                array = array[:, :, :3]
            array = np.ascontiguousarray(array, dtype=np.uint8)
        height, width = array.shape[:2]
        if (width, height) != (self.width, self.height):
            # Keep the stream stable if a source changes size. Crop is cheaper
            # than a resize and only happens on a configuration mismatch.
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            copy_h = min(height, self.height)
            copy_w = min(width, self.width)
            canvas[:copy_h, :copy_w] = array[:copy_h, :copy_w]
            array = canvas
        if self._use_i420:
            import cv2

            yuv = cv2.cvtColor(array, cv2.COLOR_RGB2YUV_I420)
            return np.ascontiguousarray(yuv).tobytes()
        return array.tobytes()

    def encode(self, image, timeout: float = 0.25) -> list[EncodedPacket]:
        """Submit one frame and return the encoded packets that are ready."""
        self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise EncoderError("encoder process is not running")
        if process.poll() is not None:
            self._error = self._stderr or self._error or "ffmpeg exited"
            raise EncoderError(self._error)
        payload = self._prepare(image)
        try:
            process.stdin.write(payload)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._error = self._stderr or str(exc)
            raise EncoderError(f"ffmpeg input closed: {self._error}") from exc

        deadline = time.monotonic() + max(0.001, float(timeout))
        packets: list[EncodedPacket] = []
        try:
            packets.append(self._queue.get(timeout=max(0.001, deadline - time.monotonic())))
        except queue.Empty:
            pass
        while True:
            try:
                packets.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for packet in packets:
            self._packets += 1
            self._bytes += len(packet.data)
        return packets

    def wait_ready(self, timeout: float = 5.0) -> None:
        """Verify that ffmpeg did not fail during startup."""
        self.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                self._error = self._stderr or _stderr_now(process) or "ffmpeg exited"
                raise EncoderError(self._error)
            if self._stderr:
                self._error = self._stderr
                raise EncoderError(self._error)
            if process is not None and process.stdin is not None:
                # Encoders initialise on the first submitted frame. Waiting a
                # short moment catches a missing driver or unsupported mode
                # before the phone connects, without injecting a probe frame
                # into the stream.
                time.sleep(0.15)
                if process.poll() is None:
                    return
            else:
                time.sleep(0.05)
        raise EncoderError("ffmpeg did not become ready")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)


def _stderr_now(process: subprocess.Popen) -> str:
    if process.stderr is None:
        return ""
    try:
        import select

        if not select.select([process.stderr], [], [], 0)[0]:
            return ""
    except Exception:
        pass
    try:
        return process.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _normalise_backend(config: PhoneConfig) -> str:
    backend = (config.encoder or "auto").lower()
    if backend in ("cpu", "libx264"):
        return "x264"
    return backend


def select_stream_encoder(
    config: PhoneConfig, capabilities: Optional[GpuCapabilities] = None
) -> StreamSelection:
    """Choose the wire codec and the ffmpeg encoder that should produce it.

    ``phone.encoder`` is the backend knob: ``auto`` prefers NVENC and only
    falls back to CPU x264 when H.264 was explicitly requested; ``nvenc``
    never uses the CPU; ``x264`` always uses CPU H.264 and therefore rejects
    HEVC/AV1 codec selections.
    """
    codec = (config.codec or "auto").lower()
    if codec not in ("auto", "jpeg", "hevc", "av1", "h264"):
        codec = "auto"
    backend = _normalise_backend(config)
    if codec == "jpeg":
        return StreamSelection("jpeg", "", "JPEG explicitly configured")
    capabilities = capabilities or detect_gpu()
    cpu_available = bool(capabilities.ffmpeg) and CPU_ENCODER in capabilities.encoders
    if not capabilities.ffmpeg:
        cpu_reason = "ffmpeg is not available"
    elif not cpu_available:
        cpu_reason = "ffmpeg has no libx264 encoder"
    else:
        cpu_reason = ""
    if cpu_reason and capabilities.reason:
        cpu_reason = f"{cpu_reason} ({capabilities.reason})"

    if backend == "x264":
        if codec not in ("auto", "h264"):
            return StreamSelection(
                "jpeg", "", f"CPU x264 only encodes H.264, not {codec}"
            )
        if not cpu_available:
            return StreamSelection("jpeg", "", cpu_reason)
        return StreamSelection(
            "h264", CPU_ENCODER, f"CPU libx264 ({capabilities.ffmpeg})"
        )

    if backend == "nvenc":
        if not capabilities.available:
            return StreamSelection(
                "jpeg", "", capabilities.reason or "NVENC is not available"
            )
        if codec == "auto":
            chosen = capabilities.best_codec(allow_av1=True)
            if chosen == "jpeg":
                return StreamSelection("jpeg", "", "no usable NVENC encoder")
            return StreamSelection(
                chosen, CODEC_ENCODERS[chosen], capabilities.summary()
            )
        if not capabilities.supports(codec):
            return StreamSelection("jpeg", "", f"{codec}_nvenc is not available")
        return StreamSelection(codec, CODEC_ENCODERS[codec], capabilities.summary())

    # auto: NVENC first, CPU H.264 for an explicit H.264 request.
    if codec == "auto":
        chosen = capabilities.best_codec(allow_av1=True)
        if chosen != "jpeg":
            return StreamSelection(
                chosen, CODEC_ENCODERS[chosen], capabilities.summary()
            )
        return StreamSelection(
            "jpeg", "", capabilities.reason or "no NVIDIA GPU detected"
        )
    if capabilities.supports(codec):
        return StreamSelection(codec, CODEC_ENCODERS[codec], capabilities.summary())
    if codec == "h264" and cpu_available:
        return StreamSelection(
            "h264", CPU_ENCODER, f"CPU libx264; no NVENC H.264 ({capabilities.reason})"
        )
    if codec == "h264" and not cpu_available:
        return StreamSelection("jpeg", "", cpu_reason)
    return StreamSelection("jpeg", "", f"{codec}_nvenc is not available")


def choose_stream_codec(
    config: PhoneConfig, capabilities: Optional[GpuCapabilities] = None
) -> tuple[str, str]:
    """Return ``(codec, reason)`` for the configured phone stream."""
    selection = select_stream_encoder(config, capabilities)
    return selection.codec, selection.reason


def create_stream_encoder(config: PhoneConfig, width: int, height: int):
    """Create a hardware/CPU encoder or return ``None`` for JPEG."""
    capabilities = detect_gpu()
    selection = select_stream_encoder(config, capabilities)
    if selection.codec == "jpeg":
        LOGGER.info("phone video using JPEG: %s", selection.reason)
        return None
    try:
        encoder = FfmpegVideoEncoder(
            selection.codec,
            width,
            height,
            fps=config.fps,
            bitrate_mbps=config.bitrate_mbps,
            keyframe_interval_s=config.keyframe_interval_s,
            ffmpeg=capabilities.ffmpeg,
            encoder=selection.encoder,
        )
        encoder.wait_ready()
    except EncoderError as exc:
        LOGGER.warning("video encoder unavailable, falling back to JPEG: %s", exc)
        return None
    LOGGER.info(
        "phone video using %s via %s at %dx%d, %.0f Mbps (%s)",
        selection.codec.upper(),
        selection.encoder,
        width,
        height,
        config.bitrate_mbps,
        selection.reason,
    )
    return encoder


# The class used to be NVENC-only. Keep the old name importable for callers
# that pinned it before the CPU backend was added.
NvencVideoEncoder = FfmpegVideoEncoder
