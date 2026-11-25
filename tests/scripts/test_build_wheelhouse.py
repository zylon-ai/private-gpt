import hashlib
from pathlib import Path

from scripts.build_wheelhouse import rewrite_hashes

LOCKED = """\
aiohttp==3.13.3 \\
    --hash=sha256:{a}
langdetect==1.0.9 \\
    --hash=sha256:{b}
yarl==1.22.0 ; python_full_version >= '3.11' \\
    --hash=sha256:{c} \\
    --hash=sha256:{d}
""".format(a="a" * 64, b="b" * 64, c="c" * 64, d="d" * 64)


def test_sdist_hash_replaced_by_built_wheel_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "langdetect-1.0.9-py3-none-any.whl"
    wheel.write_bytes(b"built wheel")

    out = rewrite_hashes(LOCKED, {"langdetect": wheel})

    expected = hashlib.sha256(b"built wheel").hexdigest()
    assert f"langdetect==1.0.9 \\\n    --hash=sha256:{expected}\n" in out
    assert "b" * 64 not in out


def test_other_requirements_and_markers_untouched(tmp_path: Path) -> None:
    wheel = tmp_path / "langdetect-1.0.9-py3-none-any.whl"
    wheel.write_bytes(b"x")

    out = rewrite_hashes(LOCKED, {"langdetect": wheel})

    assert f"aiohttp==3.13.3 \\\n    --hash=sha256:{'a' * 64}\n" in out
    assert "yarl==1.22.0 ; python_full_version >= '3.11' \\" in out
    assert "c" * 64 in out
    assert "d" * 64 in out
