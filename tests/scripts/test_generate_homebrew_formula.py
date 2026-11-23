from scripts.generate_homebrew_formula import render_formula

CONSTRAINTS = (
    "https://wheels.privategpt.dev/constraints/private-gpt-1.1.0-constraints.txt"
)


def _formula() -> str:
    return render_formula(
        version="1.1.0",
        source_url="https://example.invalid/v1.1.0.tar.gz",
        sha256="0" * 64,
        homepage="https://github.com/zylon-ai/private-gpt",
        package_files_url="https://wheels.privategpt.dev/packages/",
        constraints_url=CONSTRAINTS,
        extra="core",
    )


def test_formula_depends_on_native_libmagic() -> None:
    # python-magic (core) loads libmagic via ctypes; macOS does not ship it.
    assert 'depends_on "libmagic"' in _formula()


def test_formula_pins_package_version_and_python() -> None:
    formula = _formula()
    assert '"private-gpt[core]==1.1.0"' in formula
    assert 'depends_on "python@3.11"' in formula


def test_formula_pins_dependencies_to_release_lock() -> None:
    assert f'--constraints "{CONSTRAINTS}"' in _formula()


def test_formula_test_block_matches_installed_wrapper_paths() -> None:
    # The wrapper is written with opt_bin (the stable symlink); asserting
    # opt_bin.realpath (the versioned Cellar path) can never match it.
    formula = _formula()
    assert "opt_bin.realpath" not in formula
    assert 'assert_match Formula["uv"].opt_bin.to_s, script' in formula
