"""Unit coverage for the MediaPipe Tasks model cache and downloader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from openvrcardboard import mediapipe_models as models


class FakeResponse:
    def __init__(self, payload: bytes, content_length: bool = True):
        self._payload = payload
        self._offset = 0
        self.headers = {}
        if content_length:
            self.headers["Content-Length"] = str(len(payload))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _spec(payload: bytes, name: str = "test_model.task") -> models.ModelSpec:
    return models.ModelSpec(
        key=name,
        filename=name,
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def test_pose_variant_lookup_is_case_insensitive_and_falls_back():
    assert models.pose_model_spec("lite") is models.POSE_LANDMARKER_LITE
    assert models.pose_model_spec(" HEAVY ") is models.POSE_LANDMARKER_HEAVY
    assert models.pose_model_spec("unknown") is models.POSE_LANDMARKER_FULL


def test_model_spec_accepts_names_and_specs():
    assert models.model_spec("hand_landmarker") is models.HAND_LANDMARKER
    spec = models.POSE_LANDMARKER_LITE
    assert models.model_spec(spec) is spec
    with pytest.raises(models.ModelUnavailable):
        models.model_spec("nope")


def test_model_urls_use_the_official_layout():
    assert models.HAND_LANDMARKER.url == (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    assert "/pose_landmarker/pose_landmarker_full/" in models.POSE_LANDMARKER_FULL.url


def test_default_model_dir_honours_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv(models.MODEL_DIR_ENV, str(tmp_path))
    assert models.default_model_dir() == tmp_path


def test_search_dirs_are_ordered_and_deduplicated(monkeypatch, tmp_path):
    override = tmp_path / "override"
    monkeypatch.setenv(models.MODEL_DIR_ENV, str(override))
    monkeypatch.setattr(models, "bundled_model_dirs", lambda: [tmp_path, override])
    dirs = models.search_dirs(tmp_path)
    assert dirs == [override, tmp_path]


def test_verify_model_checks_size_and_digest(tmp_path):
    payload = b"model-bytes"
    spec = _spec(payload)
    good = tmp_path / spec.filename
    good.write_bytes(payload)
    assert models.verify_model(good, spec)

    bad = tmp_path / "bad.task"
    bad.write_bytes(b"other")
    assert not models.verify_model(bad, spec)
    assert not models.verify_model(tmp_path / "missing.task", spec)


def test_download_model_uses_a_progress_callback_and_publishes_atomically(
    monkeypatch, tmp_path
):
    payload = b"x" * 32
    spec = _spec(payload)
    monkeypatch.setattr(
        models.urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload)
    )
    seen = []
    target = models.download_model(spec, tmp_path, progress=lambda r, t: seen.append((r, t)))

    assert target == tmp_path / spec.filename
    assert target.read_bytes() == payload
    assert seen[-1] == (len(payload), len(payload))
    assert not (tmp_path / (spec.filename + ".part")).exists()


def test_download_model_skips_an_existing_valid_file(monkeypatch, tmp_path):
    payload = b"already-there"
    spec = _spec(payload)
    target = tmp_path / spec.filename
    target.write_bytes(payload)

    def fail(*args, **kwargs):
        raise AssertionError("download should not run for a valid cache entry")

    monkeypatch.setattr(models.urllib.request, "urlopen", fail)
    assert models.download_model(spec, tmp_path) == target


def test_download_model_removes_a_corrupt_partial_file(monkeypatch, tmp_path):
    payload = b"expected"
    spec = _spec(payload)
    monkeypatch.setattr(
        models.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(b"tampered"),
    )
    with pytest.raises(models.ModelUnavailable, match="SHA-256"):
        models.download_model(spec, tmp_path)
    assert not (tmp_path / spec.filename).exists()
    assert not (tmp_path / (spec.filename + ".part")).exists()


def test_download_model_wraps_network_errors(monkeypatch, tmp_path):
    spec = _spec(b"payload")

    def fail(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr(models.urllib.request, "urlopen", fail)
    with pytest.raises(models.ModelUnavailable, match="network down"):
        models.download_model(spec, tmp_path)


def test_resolve_model_uses_a_valid_cache_entry(tmp_path):
    payload = b"cached"
    spec = _spec(payload)
    (tmp_path / spec.filename).write_bytes(payload)
    assert models.resolve_model(spec, tmp_path, allow_download=False) == (
        tmp_path / spec.filename
    )


def test_resolve_model_deletes_a_corrupt_cache_entry(tmp_path):
    spec = _spec(b"correct")
    corrupt = tmp_path / spec.filename
    corrupt.write_bytes(b"wrong")
    with pytest.raises(models.ModelUnavailable):
        models.resolve_model(spec, tmp_path, allow_download=False)
    assert not corrupt.exists()


def test_download_all_returns_every_requested_path(monkeypatch, tmp_path):
    requested = []

    def fake_download(name, target_dir, progress=None):
        requested.append(name)
        path = Path(target_dir) / f"{name}.task"
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(models, "download_model", fake_download)
    paths = models.download_all(tmp_path, ["a", "b"])
    assert requested == ["a", "b"]
    assert [path.name for path in paths] == ["a.task", "b.task"]


def test_main_list_prints_every_model(capsys):
    assert models._main(["--list"]) == 0
    output = capsys.readouterr().out
    for spec in models.MODELS.values():
        assert spec.filename in output
