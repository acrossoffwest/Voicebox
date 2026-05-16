from pathlib import Path

from system_checks import (
    Check,
    check_base_models,
    check_mic_permission,
    install_command,
    run_all,
)


def test_check_base_models_missing(tmp_path):
    c = check_base_models(tmp_path)
    assert c.key == "baseModels"
    assert c.status == "todo"
    assert "Missing" in c.detail
    assert c.action == "download_base_models"


def test_check_base_models_present(tmp_path):
    (tmp_path / "hubert_base.pt").write_bytes(b"\x00" * 1024 * 1024)
    (tmp_path / "rmvpe.pt").write_bytes(b"\x00" * 1024 * 1024)
    c = check_base_models(tmp_path)
    assert c.status == "ok"
    assert "MB" in c.detail


def test_check_mic_permission_is_todo():
    import system_checks as sc

    sc._mic_granted = None
    sc._mic_denied = False
    c = check_mic_permission()
    assert c.status == "todo"
    assert c.action == "request_mic"


def test_install_command_known_actions():
    assert install_command("install_brew") is not None
    assert install_command("install_blackhole") == "brew install blackhole-2ch"
    assert install_command("install_python310") == "brew install python@3.10"
    assert install_command("run_setup") == "./setup.sh"
    assert install_command("unknown") is None


def test_run_all_returns_five_checks(tmp_path):
    checks = run_all(tmp_path)
    keys = [c.key for c in checks]
    assert keys == [
        "homebrew",
        "python",
        "blackhole",
        "multiOutput",
        "baseModels",
    ]
    for c in checks:
        assert isinstance(c, Check)
