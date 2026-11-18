import socket
import threading

import numpy as np
import pytest

from openvrcardboard.phone_link import PhoneImuReceiver
from openvrcardboard.protocol import (
    pack_imu,
    pack_time_ping,
    unpack_time_pong,
)


def valid_imu(sequence: int, **kwargs):
    values = dict(
        timestamp=10.0,
        quaternion=[0.0, 0.0, 0.0, 1.0],
        gyroscope=[0.1, 0.2, 0.3],
        acceleration=[0.0, 0.0, 9.81],
        accuracy=3,
        screen_rotation=2,
    )
    values.update(kwargs)
    return pack_imu(sequence=sequence, **values)


def feed(receiver, sequence, now, phone_time=None, address=("192.168.1.20", 5555)):
    """Deliver one packet and flush the reorder buffer deterministically."""
    if phone_time is None:
        phone_time = now
    accepted = receiver.handle_datagram(
        valid_imu(sequence, timestamp=phone_time), address, now=now
    )
    receiver.flush(now + 1.0)
    return accepted


def test_receiver_accepts_samples_and_rejects_out_of_order():
    events = []
    receiver = PhoneImuReceiver(port=0, on_sample=events.append)
    try:
        address = ("192.168.1.20", 5555)
        assert receiver.handle_datagram(valid_imu(10, timestamp=1.0), address, now=1.0)
        receiver.flush(2.0)
        assert receiver.handle_datagram(valid_imu(9, timestamp=0.99), address, now=1.01) is False
        assert receiver.handle_datagram(valid_imu(10, timestamp=1.0), address, now=1.02) is False
        assert receiver.handle_datagram(valid_imu(11, timestamp=1.03), address, now=1.03)
        receiver.flush(2.0)

        assert [sample.sequence for sample in events] == [10, 11]
        sample = receiver.latest()
        assert sample.sequence == 11
        assert sample.source_address == "192.168.1.20:5555"
        assert sample.screen_rotation == 2
        assert sample.quaternion == pytest.approx([0.0, 0.0, 0.0, 1.0])
        assert sample.received_at == 1.03
        assert sample.timestamp == pytest.approx(1.03)
    finally:
        receiver.stop()


def test_receiver_resets_sequence_and_clock_for_new_phone():
    receiver = PhoneImuReceiver(port=0)
    try:
        feed(receiver, 500, 1.0, phone_time=10.0, address=("10.0.0.2", 1000))
        feed(receiver, 1, 1.1, phone_time=20.0, address=("10.0.0.3", 2000))
        feed(receiver, 2, 1.11, phone_time=20.01, address=("10.0.0.3", 2000))
        sample = receiver.latest()
        assert sample.source_address == "10.0.0.3:2000"
        assert sample.timestamp == pytest.approx(1.11, abs=0.02)
        assert receiver.stats(now=1.2).rate_hz > 0.0
    finally:
        receiver.stop()


def test_receiver_reorders_network_packets_and_drops_stale_ones():
    events = []
    receiver = PhoneImuReceiver(port=0, on_sample=events.append)
    address = ("10.0.0.2", 1000)
    try:
        # Sequence 2 overtakes sequence 1 on the network.  Both are still
        # emitted in sensor-time order once the buffer fills.
        assert receiver.handle_datagram(
            valid_imu(2, timestamp=10.05), address, now=1.05
        )
        assert receiver.handle_datagram(
            valid_imu(1, timestamp=10.04), address, now=1.06
        )
        assert receiver.handle_datagram(
            valid_imu(3, timestamp=10.06), address, now=1.06
        )
        receiver.flush(2.0)
        assert [sample.sequence for sample in events] == [1, 2, 3]
        assert [sample.timestamp for sample in events] == pytest.approx(
            [1.04, 1.05, 1.06]
        )
        # A late sample older than the emit watermark is discarded, not
        # delivered with an old pose.
        assert receiver.handle_datagram(
            valid_imu(4, timestamp=10.03), address, now=1.07
        ) is False
        assert receiver.stats().dropped_samples >= 1
    finally:
        receiver.stop()


def test_receiver_rejects_non_finite_and_malformed_packets():
    receiver = PhoneImuReceiver(port=0)
    try:
        address = ("10.0.0.2", 1000)
        assert not receiver.handle_datagram(valid_imu(1, quaternion=[0, 0, float("nan"), 1]), address)
        assert not receiver.handle_datagram(valid_imu(2, gyroscope=[float("inf"), 0, 0]), address)
        assert not receiver.handle_datagram(b"junk", address)
        stats = receiver.stats()
        assert stats.samples == 0
        assert stats.parse_errors >= 3
    finally:
        receiver.stop()


def test_receiver_estimates_rate():
    receiver = PhoneImuReceiver(port=0)
    try:
        address = ("10.0.0.2", 1000)
        for index in range(20):
            now = 1.0 + index * 0.01
            assert feed(receiver, index, now, phone_time=10.0 + index * 0.01,
                        address=address)
        stats = receiver.stats(now=1.19)
        assert stats.rate_hz == pytest.approx(100.0, rel=0.15)
        assert stats.samples == 20
        assert stats.last_sequence == 19
    finally:
        receiver.stop()


def test_receiver_answers_time_ping():
    receiver = PhoneImuReceiver(port=0)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.bind(("127.0.0.1", 0))
    client.settimeout(1.0)
    try:
        receiver.start()
        client.sendto(pack_time_ping(7, 123.5), ("127.0.0.1", receiver.local_port))
        data, _ = client.recvfrom(4096)
        pong = unpack_time_pong(data)
        assert pong.sequence == 7
        assert pong.client_time == pytest.approx(123.5)
        assert pong.server_time > 0.0
        assert receiver.stats().pings == 1
    finally:
        client.close()
        receiver.stop()


def test_receiver_thread_receives_sample_and_restarts():
    received = threading.Event()
    receiver = PhoneImuReceiver(port=0, on_sample=lambda sample: received.set())
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.start()
        client.sendto(valid_imu(1, timestamp=10.0), ("127.0.0.1", receiver.local_port))
        assert received.wait(1.0)
        assert receiver.stats().connected
        receiver.stop()
        receiver.start()
        assert receiver.running
    finally:
        receiver.stop()
        client.close()
