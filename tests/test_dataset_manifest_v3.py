import json

from ecgcert.data.manifest import DatasetManifest, build_wfdb_manifest


def test_external_manifest_roundtrip_and_file_verification(tmp_path):
    header = tmp_path / "record.hea"
    signal = tmp_path / "record.dat"
    header.write_text("record 12 500 2\n", encoding="utf-8")
    signal.write_bytes(b"fixture-signal")
    manifest = build_wfdb_manifest(
        cohort="fixture",
        version="1",
        source_url="https://example.invalid/fixture",
        root=tmp_path,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    loaded = DatasetManifest.from_path(path)
    assert loaded.sha256() == manifest.sha256()
    loaded.verify_files()
    signal.write_bytes(b"changed")
    try:
        loaded.verify_files()
    except ValueError as exc:
        assert "signal" in str(exc)
    else:
        raise AssertionError("changed signal must fail manifest verification")
