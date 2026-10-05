from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.seed_proxy_preferences import (
    DEFAULT_MANIFEST,
    DEFAULT_SOURCE,
    assert_held_out_disjoint,
    load_proxy_preferences,
    main,
    sha256_file,
    validate_dataset,
)


def _modified(tmp_path: Path, edit) -> Path:
    dataset = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    edit(dataset)
    target = tmp_path / "proxy.json"
    target.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    return target


def test_checked_in_proxy_manifest_has_auditable_labels() -> None:
    dataset = validate_dataset()
    assert len(dataset["pairs"]) == 8
    assert len(dataset["assets"]) == 15
    assert sum(pair["choice"] == "preferred" for pair in dataset["pairs"]) == 6
    assert sum(pair["choice"] == "tie" for pair in dataset["pairs"]) == 2
    assert dataset["evaluation"]["eligible_as_ground_truth"] is False
    assert dataset["public_source_manifest"]["sha256"] == sha256_file(DEFAULT_SOURCE)


@pytest.mark.parametrize(
    "field,value",
    [("source", "human"), ("authorized_by_user", False), ("human_observed", True)],
)
@pytest.mark.parametrize("level", ["dataset", "pair"])
def test_no_relabeling_proxy_as_human(tmp_path, field, value, level) -> None:
    def edit(dataset):
        target = dataset if level == "dataset" else dataset["pairs"][0]
        target[field] = value

    with pytest.raises(ValueError, match="provenance"):
        validate_dataset(_modified(tmp_path, edit))


def test_loading_requires_explicit_proxy_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_assistant_proxy"):
        load_proxy_preferences()


def test_opt_in_loader_preserves_source_and_content_pins(monkeypatch) -> None:
    dataset = validate_dataset()
    monkeypatch.setattr(
        "scripts.seed_proxy_preferences.validate_dataset",
        lambda *args, **kwargs: dataset,
    )
    # Image verification is mocked here; the CLI smoke separately checks real bytes.
    rows = load_proxy_preferences(allow_assistant_proxy=True)
    assert len(rows) == 8
    assert rows[0]["source"] == "assistant_proxy"
    assert rows[0]["human_observed"] is False
    assert rows[0]["dataset_sha256"] == sha256_file(DEFAULT_MANIFEST)
    assert rows[0]["left_asset"]["sha256"] == dataset["assets"][0]["sha256"]
    assert rows[1]["choice"] == "tie"
    assert rows[1]["preferred_file"] is None
    assert "user_id" not in rows[0]


def test_export_refuses_to_overwrite_existing_file(tmp_path, monkeypatch) -> None:
    dataset = validate_dataset()
    monkeypatch.setattr(
        "scripts.seed_proxy_preferences.validate_dataset",
        lambda *args, **kwargs: dataset,
    )
    output = tmp_path / "export.jsonl"
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        main(["--allow-assistant-proxy", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("confidence", [True, -0.01, 1.01, float("nan"), float("inf")])
def test_invalid_confidence_is_rejected(tmp_path, confidence) -> None:
    with pytest.raises(ValueError, match="confidence"):
        validate_dataset(
            _modified(
                tmp_path, lambda data: data["pairs"][0].update(confidence=confidence)
            )
        )


def test_tie_does_not_invent_a_winner(tmp_path) -> None:
    def edit(dataset):
        pair = dataset["pairs"][1]
        pair["preferred_file"] = pair["left_file"]

    with pytest.raises(ValueError, match="winner"):
        validate_dataset(_modified(tmp_path, edit))


def test_duplicate_pair_id_rejected(tmp_path) -> None:
    def edit(dataset):
        dataset["pairs"].append(copy.deepcopy(dataset["pairs"][0]))

    with pytest.raises(ValueError, match="duplicate pair"):
        validate_dataset(_modified(tmp_path, edit))


@pytest.mark.parametrize(
    "filename",
    ["../private.jpg", "..\\private.jpg", "C:\\private.jpg", "photo.jpg:stream"],
)
def test_unsafe_image_path_rejected(tmp_path, filename) -> None:
    with pytest.raises(ValueError, match="unsafe filename"):
        validate_dataset(
            _modified(tmp_path, lambda data: data["assets"][0].update(file=filename))
        )


def test_changed_public_manifest_fails_closed(tmp_path) -> None:
    source = tmp_path / DEFAULT_SOURCE.name
    source.write_bytes(DEFAULT_SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="content pin"):
        validate_dataset(source_manifest=source)


def test_missing_images_fail_before_export(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_proxy_preferences(allow_assistant_proxy=True, public_album=tmp_path)


def test_changed_image_bytes_fail_closed(tmp_path) -> None:
    dataset = validate_dataset()
    asset = dataset["assets"][0]
    (tmp_path / asset["file"]).write_bytes(b"not-the-public-photo")
    with pytest.raises(ValueError, match="image content pin"):
        load_proxy_preferences(allow_assistant_proxy=True, public_album=tmp_path)


def test_entire_exposed_source_corpus_excluded_from_future_holdout() -> None:
    source = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
    dataset = validate_dataset()
    annotated = {asset["sha256"] for asset in dataset["assets"]}
    unannotated = next(
        asset["expected_sha256"]
        for asset in source["images"]
        if asset["expected_sha256"] not in annotated
    )
    for digest in (next(iter(annotated)), unannotated):
        with pytest.raises(ValueError, match="overlaps"):
            assert_held_out_disjoint([digest])
    assert_held_out_disjoint(["0" * 64])


def test_disabling_evaluation_exclusion_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="evaluation exclusion"):
        validate_dataset(
            _modified(
                tmp_path,
                lambda data: data["evaluation"].update(
                    exclude_entire_source_corpus=False
                ),
            )
        )


def test_cli_cannot_export_without_opt_in(tmp_path) -> None:
    output = tmp_path / "export.jsonl"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_duplicate_json_keys_rejected(tmp_path) -> None:
    manifest = tmp_path / "duplicate.json"
    manifest.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        validate_dataset(manifest)
