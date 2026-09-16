"""Matching an already-installed model: after a download Erudi rewrites the row's `link` to the local path."""

import pytest

from erudi_eval.model_match import ModelMatchError, pick_installed

CATALOG_LINK = "lmstudio-community/Qwen3-4B-MLX-4bit"
DATA_ROOT = "/Users/u/Library/Application Support/erudi/backend/prod"
# Recorded shapes from the first real run (2026-09-15): the catalog row keeps the HF link with local 0,
# the installed row's link is the local model directory.
CATALOG = [{"id": 20, "name": "Qwen3 4B", "link": CATALOG_LINK, "local": 0, "artifact_size_bytes": 2278972619}]
INSTALLED_BY_PATH = {"id": 358, "name": "Qwen3 4B", "link": f"{DATA_ROOT}/data/models/358", "local": 1, "weights_available": True}
INSTALLED_BY_LINK = {"id": 41, "name": "Qwen3 4B", "link": CATALOG_LINK, "local": 1, "weights_available": True}
ASSISTANT = {"id": 400, "name": "erudi-eval kb", "link": f"{DATA_ROOT}/data/models/358", "local": 1, "is_attached_to_kb": 1}
OTHER = {"id": 99, "name": "Gemma 3 4B", "link": "google/gemma-3-4b-it-qat-q4_0-gguf", "local": 1, "weights_available": True}


def test_matches_on_the_catalog_link_when_the_row_kept_it():
    m = pick_installed([INSTALLED_BY_LINK, OTHER, ASSISTANT], CATALOG, CATALOG_LINK, DATA_ROOT)
    assert m.model["id"] == 41 and m.rule == "link" and m.by_name is False and m.candidates == 1


def test_matches_on_the_catalog_name_when_the_link_was_rewritten_to_the_local_path():
    m = pick_installed([INSTALLED_BY_PATH, OTHER, ASSISTANT], CATALOG, CATALOG_LINK, DATA_ROOT)
    assert m.model["id"] == 358 and m.rule == "name" and m.by_name is True
    assert m.catalog_name == "Qwen3 4B" and "name" in m.note


def test_prefers_a_row_inside_the_data_root_then_the_oldest_when_several_match():
    duplicate = {**INSTALLED_BY_PATH, "id": 360, "link": f"{DATA_ROOT}/data/models/360"}
    elsewhere = {**INSTALLED_BY_PATH, "id": 500, "link": "/tmp/somewhere/Qwen3-4B"}
    m = pick_installed([elsewhere, duplicate, INSTALLED_BY_PATH], CATALOG, CATALOG_LINK, DATA_ROOT)
    assert m.model["id"] == 358 and m.candidates == 3 and "3 installed rows" in m.note


def test_ignores_kb_assistants_missing_weights_and_other_models():
    rows = [ASSISTANT, OTHER, {**INSTALLED_BY_PATH, "weights_available": False}]
    with pytest.raises(ModelMatchError, match="weights"):
        pick_installed(rows, CATALOG, CATALOG_LINK, DATA_ROOT)
    assert pick_installed([ASSISTANT, OTHER], CATALOG, CATALOG_LINK, DATA_ROOT) is None


def test_downloading_row_and_unknown_catalog_link():
    downloading = {**INSTALLED_BY_PATH, "local": 2}
    with pytest.raises(ModelMatchError, match="being downloaded"):
        pick_installed([downloading], CATALOG, CATALOG_LINK, DATA_ROOT)
    # Without the catalog row the name is unknown: only the link rule can match, and it does not here.
    assert pick_installed([INSTALLED_BY_PATH], [], CATALOG_LINK, DATA_ROOT) is None
