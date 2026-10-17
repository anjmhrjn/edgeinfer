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


def test_eval_refuses_a_subset_outside_the_manifest(capsys):
    # sens_val100 isn't part of golden_val32: its scores would silently cover fewer images.
    rc = main(["eval", "--model", "segformer_b0", "--manifest", "golden_val32",
               "--subset", "sens_val100"])  # fmt: skip
    assert rc == 2
    assert "aren't in golden_val32" in capsys.readouterr().out
