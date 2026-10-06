from pathlib import Path

from scripts.start_demo import demo_environment
from scripts.start_demo import default_web_dist


def test_fresh_checkout_uses_bundled_frontend(tmp_path):
    assert default_web_dist(tmp_path) == tmp_path / "ai/web_dist"
    preview = tmp_path / ".norma/demo-web-dist"
    preview.mkdir(parents=True)
    assert default_web_dist(tmp_path) == tmp_path / "ai/web_dist"
    (preview / "index.html").write_text("<html></html>")
    assert default_web_dist(tmp_path) == preview


def test_defaults_are_local_and_record_only():
    env = demo_environment(Path("test-data"), Path("test-dist"), environ={})
    assert env["NORMA_VLM_PROVIDER"] == "local"
    assert env["NORMA_PREFERENCE_MODE"] == "record-only"
    assert env["NORMA_DATA_DIR"] == str(Path("test-data").resolve())


def test_explicit_provider_and_other_settings_preserved():
    original = {
        "NORMA_VLM_PROVIDER": "openai-compatible",
        "NORMA_VLM_API_KEY": "test-secret",
        "NORMA_MODEL_CACHE_DIR": "chosen-cache",
    }
    env = demo_environment(Path("test-data"), Path("test-dist"), environ=original)
    assert all(env[k] == v for k, v in original.items())
    override = demo_environment(Path("test-data"), Path("test-dist"), "local", original)
    assert override["NORMA_VLM_PROVIDER"] == "local"
    assert original["NORMA_VLM_PROVIDER"] == "openai-compatible"
