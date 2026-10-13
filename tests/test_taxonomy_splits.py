import json
from types import SimpleNamespace

import numpy as np
import pytest

from bird_desktop.data.bird_names import BirdNamesLoader
from bird_desktop.data.taxonomy_splits import (
    TAXONOMY_REVISION, YELLOW_WARBLER, refine, review_candidates, splits_for_labels,
)
from bird_desktop.inference.cache import PredictionCache
from bird_desktop.inference.corrections import ManualCorrectionStore
from bird_desktop.inference.pipeline import BirdOnnxPipeline


@pytest.fixture
def labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("species_id,scientific_name,name_en\n9152,Setophaga petechia,Mangrove Yellow Warbler\n")
    return path


@pytest.mark.parametrize("lat,lon,day,preferred", [
    (36.24, -112.69, 240, "Setophaga aestiva"),  # Havasupai
    (40.23, -111.73, None, "Setophaga aestiva"),
    (-0.7, -90.4, None, "Setophaga petechia"),  # Galapagos resident
    (18.2, -66.5, 180, "Setophaga petechia"),  # Caribbean summer
    (18.2, -66.5, 30, None),  # Winter overlap
    (18.2, -66.5, None, None),
    (20.9, -86.8, 180, None),  # Unsupported mainland/coast
    (None, None, 180, None), (36.2, None, 180, None),
    (float("nan"), -112.7, 180, None), (91, 0, 180, None),
    (0, 0, 180, None), (36, -181, 180, None),
])
def test_geographic_suggestions_never_invent_species_confidence(lat, lon, day, preferred):
    original = {"species_id": "9152", "confidence": 1.0}
    result = refine(original, YELLOW_WARBLER, lat, lon, day)
    assert original == {"species_id": "9152", "confidence": 1.0}
    assert result["species_id"] == "9152"
    assert result["confidence"] == 1.0
    refinement = result["taxonomy_refinement"]
    assert refinement["model_confidence"] == 1.0
    assert refinement["requires_confirmation"] is True
    assert refinement["status"] == ("suggested" if preferred else "unresolved")
    suggestions = refinement["suggestions"]
    assert len(suggestions) == 2
    assert all(item["confidence"] is None for item in suggestions)
    assert [item["scientific_name"] for item in suggestions if item["preferred"]] == ([preferred] if preferred else [])
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_label_coverage_and_manual_correction_names(labels):
    assert splits_for_labels(labels) == {"9152": YELLOW_WARBLER}
    names = BirdNamesLoader(labels)
    assert names.get("9152").name == "Yellow Warbler (Northern / Mangrove)"
    assert names.get("taxonomy:Setophaga aestiva").name == "Northern Yellow Warbler"
    assert names.get("taxonomy:Setophaga petechia").name == "Mangrove Yellow Warbler"


def test_future_split_aware_model_is_untouched(tmp_path):
    path = tmp_path / "future.csv"
    path.write_text("species_id,scientific_name,name_en\n1,Setophaga petechia,Mangrove Yellow Warbler\n2,Setophaga aestiva,Northern Yellow Warbler\n")
    assert splits_for_labels(path) == {}
    assert list(BirdNamesLoader(path).all()) == ["1", "2"]
    assert BirdNamesLoader(path).get("1").name == "Mangrove Yellow Warbler"


def test_pipeline_applies_context_for_each_row_and_preserves_order(labels):
    pipeline = BirdOnnxPipeline.__new__(BirdOnnxPipeline)
    pipeline.class_ids = ["9152", "42"]
    pipeline.taxonomy_splits = splits_for_labels(labels)
    pipeline._run_classifier = lambda feeds: (
        np.array([[1., .1], [.9, .05]], dtype=np.float32), np.array([[0, 1], [0, 1]]))
    results = pipeline._classifier_results_from_feeds({
        "lat": np.array([36.24, -0.7]), "lon": np.array([-112.69, -90.4]),
        "day": np.array([240, np.nan]),
    })
    assert results[0][0]["taxonomy_refinement"]["suggestions"][0]["scientific_name"] == "Setophaga aestiva"
    assert results[1][0]["taxonomy_refinement"]["suggestions"][0]["scientific_name"] == "Setophaga petechia"
    assert results[0][1] == {"species_id": "42", "confidence": float(np.float32(.1))}
    assert pipeline.class_ids == ["9152", "42"]


def test_review_candidates_preserve_raw_score_and_do_not_autoselect_unresolved():
    original = {"species_id": "9152", "confidence": .99}
    rows = review_candidates(refine(original, YELLOW_WARBLER), "ignored", "ignored")
    assert len(rows) == 2
    assert all(row["score"] == .99 and row["species_confidence"] is None and row["auto_select"] is False for row in rows)
    regular = {"species_id": "42", "confidence": .7}
    assert review_candidates(regular, "Crow", "Corvus") == [dict(common_name="Crow", scientific_name="Corvus", score=.7)]


def test_prediction_fingerprint_includes_taxonomy_revision(tmp_path):
    for name in ("detector.onnx", "classifier.onnx", "labels.csv"):
        (tmp_path / name).write_bytes(b"test")
    config = SimpleNamespace(detector_model=tmp_path / "detector.onnx", classifier_model=tmp_path / "classifier.onnx",
                             labels_path=tmp_path / "labels.csv", detection_threshold=.5, classification_top_k=5, crop_size=224)
    assert json.loads(PredictionCache.model_fingerprint(config))["taxonomyRevision"] == TAXONOMY_REVISION


def test_manual_confirmation_keeps_raw_refinement_separate():
    original = refine({"species_id": "9152", "confidence": 1.0}, YELLOW_WARBLER)
    bird = {"classification": [original]}
    store = ManualCorrectionStore.__new__(ManualCorrectionStore)
    store._apply_to_bird(bird, "taxonomy:Setophaga aestiva")
    assert bird["classification"][0]["species_id"] == "taxonomy:Setophaga aestiva"
    assert bird["classification"][0]["manual"] is True
    assert "taxonomy_refinement" not in bird["classification"][0]
    assert bird["manualCorrection"]["originalClassification"] == original
