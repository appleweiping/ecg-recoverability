from pathlib import Path
import hashlib
import re


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "cpu.lock.txt": "bc5534f459af61759abe6e3c640553d266d4d58f73d2cac404990584d7704ed9",
    "gpu.lock.txt": "fbe43187cea8667241409d33e0378f4cf937ffb4804a2f2182acd58d1d0efd2e",
}
REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)\s*\\$", re.MULTILINE)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _direct(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        values[_normalized(name)] = version
    return values


def _locked(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    matches = list(REQUIREMENT.finditer(text))
    values = {_normalized(match.group(1)): match.group(2) for match in matches}
    assert len(values) == len(matches), "lock contains duplicate package entries"
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        assert "--hash=sha256:" in text[match.end():end], match.group(1)
    return values, text


def test_complete_hash_locked_cpu_and_gpu_environments() -> None:
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.11.2"
    for stem in ("cpu", "gpu"):
        input_path = ROOT / "environments" / f"{stem}.in"
        lock_path = ROOT / "environments" / f"{stem}.lock.txt"
        direct = _direct(input_path)
        locked, text = _locked(lock_path)
        assert set(direct) < set(locked), "lock must include transitive dependencies"
        assert {key: locked[key] for key in direct} == direct
        assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == EXPECTED_HASHES[
            lock_path.name
        ]
        if stem == "gpu":
            assert "--extra-index-url https://download.pytorch.org/whl/cu128" in text
            assert locked["torch"] == "2.8.0+cu128"
        else:
            assert locked["torch"] == "2.8.0"
