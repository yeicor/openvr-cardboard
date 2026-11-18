"""Quaternion and small vector helpers.

Quaternions are numpy arrays in ``(x, y, z, w)`` order, matching OpenVR and the
wire protocol. The functions are intentionally dependency light: numpy only.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    length = float(np.linalg.norm(q))
    if not math.isfinite(length) or length < _EPS:
        return IDENTITY.copy()
    return q / length


def conjugate(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def inverse(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    length_sq = float(np.dot(q, q))
    if length_sq < _EPS:
        return IDENTITY.copy()
    return conjugate(q) / length_sq


def multiply(a, b) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float64).reshape(4)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def multiply_many(*quaternions) -> np.ndarray:
    result = IDENTITY.copy()
    for q in quaternions:
        result = multiply(result, q)
    return result


def rotate(q, v) -> np.ndarray:
    """Rotate a 3-vector by quaternion ``q``."""
    q = normalize(q)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    qv = q[:3]
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (q[3] * uv + uuv)


def from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if length < _EPS or not math.isfinite(angle_rad):
        return IDENTITY.copy()
    axis = axis / length
    half = float(angle_rad) * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def from_rotvec(rotvec) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < _EPS:
        return IDENTITY.copy()
    return from_axis_angle(rotvec / angle, angle)


def to_rotvec(q) -> np.ndarray:
    q = normalize(q)
    if q[3] < 0.0:
        q = -q
    w = max(-1.0, min(1.0, float(q[3])))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-8:
        return np.zeros(3)
    return q[:3] * (angle / s)


def from_euler_zyx_deg(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Yaw (Y), then pitch (X), then roll (Z), in degrees."""
    qy = from_axis_angle((0.0, 1.0, 0.0), math.radians(yaw_deg))
    qx = from_axis_angle((1.0, 0.0, 0.0), math.radians(pitch_deg))
    qz = from_axis_angle((0.0, 0.0, 1.0), math.radians(roll_deg))
    return multiply_many(qy, qx, qz)


def to_euler_zyx_deg(q) -> tuple[float, float, float]:
    """Return ``(yaw, pitch, roll)`` in degrees.

    Inverse of :func:`from_euler_zyx_deg`: yaw is around +Y, pitch around +X
    and roll around +Z, applied as ``R_y(yaw) * R_x(pitch) * R_z(roll)``.
    """
    m = to_matrix(q)
    sin_pitch = max(-1.0, min(1.0, -float(m[1, 2])))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) < 0.999999:
        yaw = math.atan2(float(m[0, 2]), float(m[2, 2]))
        roll = math.atan2(float(m[1, 0]), float(m[1, 1]))
    else:
        # Gimbal lock: yaw and roll describe the same rotation; put it all
        # into yaw and leave roll at zero.
        if sin_pitch > 0.0:
            yaw = math.atan2(float(m[0, 1]), float(m[0, 0]))
        else:
            yaw = math.atan2(-float(m[0, 1]), float(m[0, 0]))
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def from_matrix(matrix) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("expected a 3x3 matrix")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(0.0, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(0.0, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return normalize(q)


def to_matrix(q) -> np.ndarray:
    x, y, z, w = normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def slerp(a, b, t: float) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    t = max(0.0, min(1.0, float(t)))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return normalize(a + t * (b - a))
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return normalize(s0 * a + s1 * b)


def angle_between(a, b) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def yaw_only(q) -> np.ndarray:
    x, y, z, w = normalize(q)
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
    return from_axis_angle((0.0, 1.0, 0.0), yaw)


def project_on_plane(q, normal=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Remove rotation around ``normal`` from a quaternion (best effort)."""
    n = normalize_axis(normal)
    v = rotate(q, n)
    v = v - n * float(np.dot(v, n))
    if float(np.linalg.norm(v)) < 1e-8:
        return IDENTITY.copy()
    tilt = from_two_vectors(n, v / np.linalg.norm(v))
    return tilt


def normalize_axis(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(v))
    return v / length if length > _EPS else np.array([0.0, 0.0, 1.0])


def from_two_vectors(a, b) -> np.ndarray:
    a = normalize_axis(a)
    b = normalize_axis(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.999999:
        return IDENTITY.copy()
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return from_axis_angle(axis, math.pi)
    axis = np.cross(a, b)
    angle = math.acos(dot)
    return from_axis_angle(axis, angle)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def apply_deadzone(value: float, deadzone: float) -> float:
    if deadzone <= 0.0:
        return clamp(value, -1.0, 1.0)
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / (1.0 - deadzone)
    return math.copysign(clamp(scaled, 0.0, 1.0), value)


# Camera optical frame (x right, y down, z forward) -> view frame
# (x right, y up, z backward, i.e. OpenVR/HMD convention).
OPTICAL_TO_VIEW = from_axis_angle((1.0, 0.0, 0.0), math.pi)


def camera_optical_to_play(camera_position, camera_rotation, optical_position, optical_rotation):
    """Transform a pose expressed in OpenCV camera coordinates to play space.

    ``camera_rotation`` is the orientation of the camera as if it were an HMD:
    looking down -Z with +Y up. Pass the optical frame rotation (from solvePnP)
    and this function applies the one-time optical-to-view axis conversion.
    """
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    camera_rotation = normalize(camera_rotation)
    optical_position = np.asarray(optical_position, dtype=np.float64).reshape(3)
    pose_rotation = multiply_many(camera_rotation, OPTICAL_TO_VIEW, optical_rotation)
    pose_position = camera_position + rotate(camera_rotation, rotate(OPTICAL_TO_VIEW, optical_position))
    return pose_position, normalize(pose_rotation)


def euler_deg_to_quaternion(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    return from_euler_zyx_deg(yaw_deg, pitch_deg, roll_deg)
