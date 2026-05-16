from pathlib import Path

from models_manager import (
    VoiceModel,
    accept_drop,
    list_voice_models,
    remove_model,
)


def test_list_voice_models_empty(tmp_path):
    assert list_voice_models(tmp_path) == []


def test_list_voice_models_picks_up_pth_and_index(tmp_path):
    voice = tmp_path / "my_voice"
    voice.mkdir()
    (voice / "my_voice.pth").write_bytes(b"\x00" * 4096)
    (voice / "my_voice.index").write_bytes(b"\x00" * 2048)

    models = list_voice_models(tmp_path)
    assert len(models) == 1
    m = models[0]
    assert isinstance(m, VoiceModel)
    assert m.name == "my_voice"
    assert m.has_pth and m.has_index
    assert m.full
    assert m.size_label.endswith(" MB")
    assert m.files_label == ".pth + .index"


def test_list_voice_models_pth_only(tmp_path):
    voice = tmp_path / "alt"
    voice.mkdir()
    (voice / "alt.pth").write_bytes(b"\x00" * 1024)
    models = list_voice_models(tmp_path)
    assert len(models) == 1
    assert models[0].files_label == ".pth only"
    assert not models[0].full


def test_remove_model(tmp_path):
    voice = tmp_path / "rm_me"
    voice.mkdir()
    (voice / "rm_me.pth").write_bytes(b"\x00" * 100)
    remove_model("rm_me", tmp_path)
    assert not voice.exists()


def test_accept_drop_pth_creates_subdir(tmp_path):
    rvc = tmp_path / "rvc"
    rvc.mkdir()
    src = tmp_path / "alpha.pth"
    src.write_bytes(b"\x00" * 100)

    moved = accept_drop([src], rvc)
    assert len(moved) == 1
    assert moved[0] == rvc / "alpha" / "alpha.pth"
    assert moved[0].is_file()
    assert not src.exists()


def test_accept_drop_index_pairs_with_existing_voice(tmp_path):
    rvc = tmp_path / "rvc"
    voice = rvc / "beta"
    voice.mkdir(parents=True)
    (voice / "beta.pth").write_bytes(b"\x00")

    src = tmp_path / "beta.index"
    src.write_bytes(b"\x00" * 50)
    moved = accept_drop([src], rvc)
    assert moved == [voice / "beta.index"]
    assert (voice / "beta.index").is_file()


def test_accept_drop_ignores_unrelated_files(tmp_path):
    rvc = tmp_path / "rvc"
    rvc.mkdir()
    src = tmp_path / "readme.txt"
    src.write_bytes(b"hi")
    assert accept_drop([src], rvc) == []
    assert src.exists()  # not moved
