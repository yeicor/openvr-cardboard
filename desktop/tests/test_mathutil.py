import math

import numpy as np
import pytest

from openvrcardboard.mathutil import (
    IDENTITY,
    angle_between,
    apply_deadzone,
    conjugate,
    from_axis_angle,
    from_euler_zyx_deg,
    from_matrix,
    from_rotvec,
    from_two_vectors,
    inverse,
    multiply,
    normalize,
    rotate,
    slerp,
    to_euler_zyx_deg,
    to_matrix,
    to_rotvec,
)


def test_quaternion_identity_and_inverse():
    q = normalize([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(multiply(q, IDENTITY), q)
    assert np.allclose(multiply(q, inverse(q)), IDENTITY, atol=1e-9)
    assert np.allclose(multiply(q, conjugate(q)), IDENTITY, atol=1e-9)


def test_rotate_matches_matrix():
    q = from_euler_zyx_deg(35.0, -20.0, 12.0)
    vector = np.array([1.0, 2.0, 3.0])
    assert rotate(q, vector) == pytest.approx(to_matrix(q) @ vector)


def test_matrix_roundtrip():
    q = from_euler_zyx_deg(-70.0, 25.0, 40.0)
    result = from_matrix(to_matrix(q))
    assert angle_between(q, result) < 1e-6


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (0.0, 45.0, 0.0),
        (0.0, 0.0, -30.0),
        (35.0, -20.0, 12.0),
        (-170.0, 60.0, 5.0),
    ],
)
def test_euler_roundtrip(yaw, pitch, roll):
    q = from_euler_zyx_deg(yaw, pitch, roll)
    result = to_euler_zyx_deg(q)
    assert result[0] == pytest.approx(yaw, abs=1e-6)
    assert result[1] == pytest.approx(pitch, abs=1e-6)
    assert result[2] == pytest.approx(roll, abs=1e-6)


def test_euler_gimbal_lock_keeps_pitch():
    q = from_euler_zyx_deg(30.0, 90.0, 0.0)
    yaw, pitch, roll = to_euler_zyx_deg(q)
    assert pitch == pytest.approx(90.0)
    assert roll == pytest.approx(0.0)
    rebuilt = from_euler_zyx_deg(yaw, pitch, roll)
    assert angle_between(q, rebuilt) < 1e-6


def test_rotvec_roundtrip():
    rotvec = np.array([0.3, -0.4, 0.5])
    q = from_rotvec(rotvec)
    assert to_rotvec(q) == pytest.approx(rotvec)
    assert angle_between(q, IDENTITY) == pytest.approx(np.linalg.norm(rotvec))


def test_slerp_handles_shortest_path():
    a = from_axis_angle((0.0, 1.0, 0.0), math.radians(170.0))
    b = from_axis_angle((0.0, 1.0, 0.0), math.radians(-170.0))
    middle = slerp(a, b, 0.5)
    yaw = to_euler_zyx_deg(middle)[0]
    assert abs(abs(yaw) - 180.0) < 1e-6


def test_from_two_vectors_and_deadzone():
    q = from_two_vectors((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    assert rotate(q, (0.0, 1.0, 0.0)) == pytest.approx([0.0, 0.0, -1.0])
    assert apply_deadzone(0.05, 0.1) == 0.0
    assert apply_deadzone(-1.0, 0.1) == -1.0
    assert apply_deadzone(0.55, 0.1) == pytest.approx(0.5)
