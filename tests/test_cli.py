import pytest

from edgeinfer.cli import main


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_a_crash_is_not_a_failed_check(tmp_path, capsys):
    """Exit 1 means "checks failed, report written"; an exception must not look like it."""
    from edgeinfer.cli import EXIT_CRASH

    missing = tmp_path / "missing.onnx"
    assert main(["validate", "--model", "segformer_b0", "--artifact", str(missing)]) == EXIT_CRASH
    assert "Traceback" in capsys.readouterr().err
