"""Remote base-model loads must share one immutable Hub revision."""

from types import SimpleNamespace

import pytest

from llm_grading.models.loader import load_model


@pytest.mark.parametrize("requested_revision", ["main", "release", "a" * 40])
def test_remote_revision_is_pinned_without_config_commit_attribute(
    tmp_path, monkeypatch, requested_revision
):
    import transformers
    import transformers.utils

    commit = "a" * 40
    resolved = tmp_path / "snapshots" / commit / "config.json"
    revisions = []

    def cached_file(name, filename, **kwargs):
        assert kwargs["revision"] == requested_revision
        return str(resolved)

    def base_config(name, **kwargs):
        revisions.append(kwargs["revision"])
        return SimpleNamespace(model_type="gpt2")

    def tokenizer(name, **kwargs):
        revisions.append(kwargs["revision"])
        return SimpleNamespace(pad_token_id=0)

    def model(name, **kwargs):
        revisions.append(kwargs["revision"])
        return SimpleNamespace(to=lambda device: None)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", base_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model)
    # Import lazy Auto modules before replacing their exported cache function.
    monkeypatch.setattr(transformers.utils, "cached_file", cached_file)
    bundle = load_model(
        {"model": {"name_or_path": "example/model", "revision": requested_revision}},
        training=True,
    )
    assert bundle["revision"] == commit
    assert revisions == [commit, commit, commit]


def test_remote_revision_resolution_failure_stops_model_load(tmp_path, monkeypatch):
    import transformers.utils

    monkeypatch.setattr(
        transformers.utils, "cached_file", lambda *a, **kw: str(tmp_path / "config.json")
    )
    with pytest.raises(ValueError, match="immutable.*revision"):
        load_model({"model": {"name_or_path": "example/model"}}, training=True)
