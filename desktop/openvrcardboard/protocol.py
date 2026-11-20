"""Wire protocol shared by the desktop app, the OpenVR driver and the phone.

See docs/PROTOCOL.md for the byte-level specification. This module deliberately
uses only the standard library so it can be tested without OpenCV or Qt.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Iterable, Sequence

MAGIC = b"OVRC"
VERSION = 1

HEADER = struct.Struct("<4sBBHI")
DEVICE_POSE = struct.Struct("<BBH13f")
CONTROLLER_INPUT = struct.Struct("<B3xI7f")
HAPTIC = struct.Struct("<B3x3f")
STATUS = struct.Struct("<5I")
IMU = struct.Struct("<d4f3f3fB3x")
TIME_PING = struct.Struct("<d")
TIME_PONG = struct.Struct("<dd")
VIDEO_FRAME = struct.Struct("<BBHHHId")

# Phone IMU and TIME_PING packets carry this many trailing bytes when a
# pairing token is configured: the first 16 bytes of SHA-256(token).  It is a
# fixed-size authenticator, so the token itself never travels over UDP.
AUTH_SIZE = 16

assert HEADER.size == 12
assert DEVICE_POSE.size == 56
assert CONTROLLER_INPUT.size == 36
assert HAPTIC.size == 16
assert STATUS.size == 20
assert IMU.size == 52
assert VIDEO_FRAME.size == 20


class ProtocolError(ValueError):
    """Raised when a datagram or frame is malformed."""


class PacketType(IntEnum):
    STATE = 0x01
    HAPTIC = 0x02
    STATUS = 0x03
    IMU = 0x20
    TIME_PING = 0x21
    TIME_PONG = 0x22


class DeviceFlags(IntFlag):
    POSE_VALID = 1 << 0
    CONNECTED = 1 << 1
    ORIENTATION_ONLY = 1 << 2


class Buttons(IntFlag):
    TRIGGER_CLICK = 1 << 0
    GRIP_CLICK = 1 << 1
    APPLICATION_MENU = 1 << 2
    SYSTEM = 1 << 3
    A = 1 << 4
    B = 1 << 5
    TRACKPAD_CLICK = 1 << 6
    JOYSTICK_CLICK = 1 << 7
    TRACKPAD_TOUCH = 1 << 8
    TRIGGER_TOUCH = 1 << 9
    JOYSTICK_TOUCH = 1 << 10
    GRIP_TOUCH = 1 << 11
    X = 1 << 12
    Y = 1 << 13
    DPAD_UP = 1 << 14
    DPAD_DOWN = 1 << 15
    DPAD_LEFT = 1 << 16
    DPAD_RIGHT = 1 << 17
    SELECT = 1 << 18
    START = 1 << 19


class DeviceId(IntEnum):
    HMD = 0
    LEFT = 1
    RIGHT = 2
    TRACKER_BASE = 100


class VideoLayout(IntEnum):
    MONO = 0
    SBS = 1
    TOP_BOTTOM = 2


@dataclass(slots=True)
class DevicePose:
    device_id: int
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flags: DeviceFlags = DeviceFlags.POSE_VALID | DeviceFlags.CONNECTED

    def __post_init__(self) -> None:
        if not 0 <= int(self.device_id) <= 255:
            raise ValueError("device_id out of range")


@dataclass(slots=True)
class ControllerInput:
    device_id: int
    buttons: Buttons = Buttons(0)
    trigger: float = 0.0
    grip: float = 0.0
    joystick_x: float = 0.0
    joystick_y: float = 0.0
    trackpad_x: float = 0.0
    trackpad_y: float = 0.0

    def __post_init__(self) -> None:
        if self.device_id not in (DeviceId.LEFT, DeviceId.RIGHT):
            raise ValueError("controller input id must be LEFT or RIGHT")


@dataclass(slots=True)
class StatePacket:
    sequence: int
    timestamp: float
    devices: list[DevicePose] = field(default_factory=list)
    inputs: list[ControllerInput] = field(default_factory=list)
    # Horizon already applied by the sender, in milliseconds.  The driver uses
    # it for ``DriverPose_t::poseTimeOffset`` instead of predicting again.
    prediction_ms: float = 0.0


@dataclass(slots=True)
class HapticEvent:
    sequence: int
    device_id: int
    amplitude: float
    frequency: float
    duration: float


@dataclass(slots=True)
class DriverStatus:
    sequence: int
    protocol_version: int
    device_count: int
    uptime_ms: int
    state_packets: int
    parse_errors: int


@dataclass(slots=True)
class PhoneImuPacket:
    sequence: int
    timestamp: float
    quaternion: tuple[float, float, float, float]
    gyroscope: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    accuracy: int
    screen_rotation: int = 0
    auth: bytes = b""


@dataclass(slots=True)
class TimePing:
    sequence: int
    client_time: float
    auth: bytes = b""


@dataclass(slots=True)
class TimePong:
    sequence: int
    client_time: float
    server_time: float


def _header(packet_type: PacketType, sequence: int) -> bytes:
    return HEADER.pack(MAGIC, VERSION, int(packet_type), 0, sequence & 0xFFFFFFFF)


def _validate_count(count: int, maximum: int, what: str) -> None:
    if not 0 <= count <= maximum:
        raise ValueError(f"{what} count out of range: {count}")


def pack_state(
    sequence: int,
    timestamp: float,
    devices: Iterable[DevicePose],
    inputs: Iterable[ControllerInput] = (),
    prediction_ms: float = 0.0,
) -> bytes:
    """Serialise a STATE packet. Counts are limited to the protocol maxima."""
    device_list = list(devices)
    input_list = list(inputs)
    _validate_count(len(device_list), 16, "device")
    _validate_count(len(input_list), 8, "input")
    horizon_ms = float(prediction_ms)
    if not math.isfinite(horizon_ms):
        horizon_ms = 0.0
    horizon_ms = min(1000.0, max(0.0, horizon_ms))

    parts = [
        _header(PacketType.STATE, sequence),
        struct.pack("<dB3xf", float(timestamp), len(device_list), horizon_ms),
    ]
    for pose in device_list:
        parts.append(
            DEVICE_POSE.pack(
                int(pose.device_id) & 0xFF,
                int(pose.flags) & 0xFF,
                0,
                *map(float, pose.position),
                *map(float, pose.rotation),
                *map(float, pose.velocity),
                *map(float, pose.angular_velocity),
            )
        )
    parts.append(struct.pack("<B3x", len(input_list)))
    for item in input_list:
        parts.append(
            CONTROLLER_INPUT.pack(
                int(item.device_id) & 0xFF,
                int(item.buttons) & 0xFFFFFFFF,
                float(item.trigger),
                float(item.grip),
                float(item.joystick_x),
                float(item.joystick_y),
                float(item.trackpad_x),
                float(item.trackpad_y),
                0.0,
            )
        )
    return b"".join(parts)


def unpack_state(data: bytes) -> StatePacket:
    if len(data) < HEADER.size + 16:
        raise ProtocolError("truncated STATE packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.STATE:
        raise ProtocolError("not a STATE packet")
    offset = HEADER.size
    timestamp, device_count, prediction_ms = struct.unpack_from("<dB3xf", data, offset)
    offset += 16
    if device_count > 16:
        raise ProtocolError("too many devices")
    needed = offset + device_count * DEVICE_POSE.size + 4
    if len(data) < needed:
        raise ProtocolError("truncated device array")
    devices: list[DevicePose] = []
    for _ in range(device_count):
        values = DEVICE_POSE.unpack_from(data, offset)
        offset += DEVICE_POSE.size
        (device_id, flags, _reserved, *floats) = values
        try:
            devices.append(
                DevicePose(
                    device_id=device_id,
                    position=tuple(floats[0:3]),
                    rotation=tuple(floats[3:7]),
                    velocity=tuple(floats[7:10]),
                    angular_velocity=tuple(floats[10:13]),
                    flags=DeviceFlags(flags),
                )
            )
        except ValueError as exc:
            raise ProtocolError(f"invalid device pose: {exc}") from exc
    if len(data) < offset + 4:
        raise ProtocolError("missing input count")
    input_count = data[offset]
    offset += 4
    if input_count > 8:
        raise ProtocolError("too many inputs")
    if len(data) < offset + input_count * CONTROLLER_INPUT.size:
        raise ProtocolError("truncated input array")
    inputs: list[ControllerInput] = []
    for _ in range(input_count):
        values = CONTROLLER_INPUT.unpack_from(data, offset)
        offset += CONTROLLER_INPUT.size
        device_id, buttons, trigger, grip, joy_x, joy_y, pad_x, pad_y, _reserved = values
        try:
            inputs.append(
                ControllerInput(
                    device_id=device_id,
                    buttons=Buttons(buttons),
                    trigger=trigger,
                    grip=grip,
                    joystick_x=joy_x,
                    joystick_y=joy_y,
                    trackpad_x=pad_x,
                    trackpad_y=pad_y,
                )
            )
        except ValueError as exc:
            raise ProtocolError(
                f"invalid controller input (device id {device_id}): {exc}"
            ) from exc
    return StatePacket(
        sequence=sequence,
        timestamp=timestamp,
        devices=devices,
        inputs=inputs,
        prediction_ms=float(prediction_ms) if math.isfinite(prediction_ms) else 0.0,
    )


def pack_haptic(sequence: int, device_id: int, amplitude: float, frequency: float, duration: float) -> bytes:
    return _header(PacketType.HAPTIC, sequence) + HAPTIC.pack(
        int(device_id) & 0xFF, float(amplitude), float(frequency), float(duration)
    )


def unpack_haptic(data: bytes) -> HapticEvent:
    if len(data) < HEADER.size + HAPTIC.size:
        raise ProtocolError("truncated HAPTIC packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.HAPTIC:
        raise ProtocolError("not a HAPTIC packet")
    device_id, amplitude, frequency, duration = HAPTIC.unpack_from(data, HEADER.size)
    return HapticEvent(sequence, device_id, amplitude, frequency, duration)


def pack_status(
    sequence: int,
    device_count: int,
    uptime_ms: int,
    state_packets: int,
    parse_errors: int,
) -> bytes:
    return _header(PacketType.STATUS, sequence) + STATUS.pack(
        VERSION, int(device_count), int(uptime_ms), int(state_packets), int(parse_errors)
    )


def unpack_status(data: bytes) -> DriverStatus:
    if len(data) < HEADER.size + STATUS.size:
        raise ProtocolError("truncated STATUS packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.STATUS:
        raise ProtocolError("not a STATUS packet")
    protocol_version, device_count, uptime_ms, state_packets, parse_errors = STATUS.unpack_from(data, HEADER.size)
    return DriverStatus(sequence, protocol_version, device_count, uptime_ms, state_packets, parse_errors)


def auth_tag(token: str) -> bytes:
    """Return the fixed-size authenticator appended to phone UDP packets.

    An empty token returns an empty tag: authentication is opt-in and the
    desktop accepts unauthenticated phone packets only when no token is set.
    """
    if not token:
        return b""
    return hashlib.sha256(str(token).encode("utf-8")).digest()[:AUTH_SIZE]


def pack_imu(
    sequence: int,
    timestamp: float,
    quaternion: Sequence[float],
    gyroscope: Sequence[float],
    acceleration: Sequence[float],
    accuracy: int,
    screen_rotation: int = 0,
    token: str = "",
) -> bytes:
    if len(quaternion) != 4 or len(gyroscope) != 3 or len(acceleration) != 3:
        raise ValueError("invalid IMU vector length")
    tail = (int(accuracy) & 0xFF) | ((int(screen_rotation) & 0x03) << 2)
    packet = _header(PacketType.IMU, sequence) + IMU.pack(
        float(timestamp),
        *map(float, quaternion),
        *map(float, gyroscope),
        *map(float, acceleration),
        tail,
    )
    return packet + auth_tag(token)


def unpack_imu(data: bytes) -> PhoneImuPacket:
    if len(data) < HEADER.size + IMU.size:
        raise ProtocolError("truncated IMU packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.IMU:
        raise ProtocolError("not an IMU packet")
    values = IMU.unpack_from(data, HEADER.size)
    (timestamp, qx, qy, qz, qw, gx, gy, gz, ax, ay, az, tail) = values
    auth_offset = HEADER.size + IMU.size
    return PhoneImuPacket(
        sequence=sequence,
        timestamp=timestamp,
        quaternion=(qx, qy, qz, qw),
        gyroscope=(gx, gy, gz),
        acceleration=(ax, ay, az),
        accuracy=tail & 0x03,
        screen_rotation=(tail >> 2) & 0x03,
        auth=bytes(data[auth_offset : auth_offset + AUTH_SIZE]),
    )


def pack_time_ping(sequence: int, client_time: float, token: str = "") -> bytes:
    return (
        _header(PacketType.TIME_PING, sequence)
        + TIME_PING.pack(float(client_time))
        + auth_tag(token)
    )


def unpack_time_ping(data: bytes) -> TimePing:
    if len(data) < HEADER.size + TIME_PING.size:
        raise ProtocolError("truncated TIME_PING packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.TIME_PING:
        raise ProtocolError("not a TIME_PING packet")
    (client_time,) = TIME_PING.unpack_from(data, HEADER.size)
    auth_offset = HEADER.size + TIME_PING.size
    return TimePing(
        sequence=sequence,
        client_time=client_time,
        auth=bytes(data[auth_offset : auth_offset + AUTH_SIZE]),
    )


def pack_time_pong(sequence: int, client_time: float, server_time: float) -> bytes:
    return _header(PacketType.TIME_PONG, sequence) + TIME_PONG.pack(float(client_time), float(server_time))


def unpack_time_pong(data: bytes) -> TimePong:
    if len(data) < HEADER.size + TIME_PONG.size:
        raise ProtocolError("truncated TIME_PONG packet")
    magic, version, packet_type, _flags, sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION or packet_type != PacketType.TIME_PONG:
        raise ProtocolError("not a TIME_PONG packet")
    client_time, server_time = TIME_PONG.unpack_from(data, HEADER.size)
    return TimePong(sequence=sequence, client_time=client_time, server_time=server_time)


def parse_packet(data: bytes):
    """Return ``(PacketType, unpacked_object)`` for any known UDP packet."""
    if len(data) < HEADER.size:
        raise ProtocolError("packet shorter than header")
    magic, version, packet_type, _flags, _sequence = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ProtocolError("bad magic")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    try:
        kind = PacketType(packet_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown packet type {packet_type}") from exc
    if kind == PacketType.STATE:
        return kind, unpack_state(data)
    if kind == PacketType.HAPTIC:
        return kind, unpack_haptic(data)
    if kind == PacketType.STATUS:
        return kind, unpack_status(data)
    if kind == PacketType.IMU:
        return kind, unpack_imu(data)
    if kind == PacketType.TIME_PING:
        return kind, unpack_time_ping(data)
    if kind == PacketType.TIME_PONG:
        return kind, unpack_time_pong(data)
    raise ProtocolError("unhandled packet type")


VIDEO_CODEC_JPEG = 0
VIDEO_CODEC_HEVC = 1
VIDEO_CODEC_AV1 = 2
VIDEO_CODEC_H264 = 3


def video_codec_value(codec: str) -> int:
    return {
        "jpeg": VIDEO_CODEC_JPEG,
        "hevc": VIDEO_CODEC_HEVC,
        "av1": VIDEO_CODEC_AV1,
        "h264": VIDEO_CODEC_H264,
    }.get((codec or "jpeg").lower(), VIDEO_CODEC_JPEG)


def video_codec_name(value: int) -> str:
    return {
        VIDEO_CODEC_JPEG: "jpeg",
        VIDEO_CODEC_HEVC: "hevc",
        VIDEO_CODEC_AV1: "av1",
        VIDEO_CODEC_H264: "h264",
    }.get(int(value), "jpeg")


def pack_video_frame(
    sequence: int,
    timestamp: float,
    layout: VideoLayout | int,
    width: int,
    height: int,
    payload: bytes,
    codec: str = "jpeg",
    keyframe: bool = True,
    config: bool = False,
) -> bytes:
    if not 0 <= width <= 0xFFFF or not 0 <= height <= 0xFFFF:
        raise ValueError("invalid video dimensions")
    flags = video_codec_value(codec) & 0x07
    if keyframe:
        flags |= 0x08
    if config:
        flags |= 0x10
    return (
        VIDEO_FRAME.pack(
            0x01,
            int(layout),
            width,
            height,
            flags,
            sequence & 0xFFFFFFFF,
            float(timestamp),
        )
        + payload
    )


def unpack_video_frame(data: bytes) -> tuple[int, int, int, int, float, bytes]:
    """Return the legacy tuple ``(layout, width, height, sequence, time, payload)``."""
    layout, width, height, sequence, timestamp, payload, _codec, _key, _config = (
        unpack_video_frame_details(data)
    )
    return layout, width, height, sequence, timestamp, payload


def unpack_video_frame_details(
    data: bytes,
) -> tuple[int, int, int, int, float, bytes, str, bool, bool]:
    if len(data) < VIDEO_FRAME.size:
        raise ProtocolError("truncated video frame")
    kind, layout, width, height, flags, sequence, timestamp = VIDEO_FRAME.unpack_from(
        data, 0
    )
    if kind != 0x01:
        raise ProtocolError(f"unknown video frame kind {kind}")
    return (
        layout,
        width,
        height,
        sequence,
        timestamp,
        data[VIDEO_FRAME.size :],
        video_codec_name(flags & 0x07),
        bool(flags & 0x08),
        bool(flags & 0x10),
    )
