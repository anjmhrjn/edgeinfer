import pytest

from edgeinfer.cli import main


def test_cli_requires_a_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err
