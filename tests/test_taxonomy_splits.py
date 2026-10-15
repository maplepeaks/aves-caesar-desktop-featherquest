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


@pytest.mark.parametrize("lat,lon,day,preferred", [
    (40.23, -111.73, 146, True),  # Utah, May 25, 2024 (leap year)
    (40.23, -111.73, 180, True),
    (40.23, -111.73, None, False),
    (40.23, -111.73, 30, False),
    (40.23, -111.73, 280, False),
    (None, None, 146, False),
    (float('nan'), -111.73, 146, False),
    (40.23, -111.73, float('inf'), False),
    (40.23, -111.73, 367, False),
    (40.23, -111.73, 146.5, False),
    (40, -105, 146, False),  # Contact region; unsupported
    (20, -100, 146, False),  # Mexico migration/winter overlap
    (40, -75, 146, False),  # No Eastern inference rule
])
def test_warbling_vireo_context_and_original_evidence(lat, lon, day, preferred):
    from bird_desktop.data.taxonomy_splits import WARBLING_VIREO
    original = dict(species_id='10573', confidence=.986)
    result = refine(original, WARBLING_VIREO, lat, lon, day)
    refinement = result['taxonomy_refinement']
    assert result['confidence'] == refinement['model_confidence'] == .986
    assert result['species_id'] == refinement['original_species_id'] == '10573'
    assert refinement['requires_confirmation'] is True
    assert refinement['status'] == ('suggested' if preferred else 'unresolved')
    assert {t['scientific_name'] for t in refinement['suggestions']} == {'Vireo gilvus', 'Vireo swainsoni'}
    assert all(t['confidence'] is None for t in refinement['suggestions'])
    rows = review_candidates(result, 'ignored', 'ignored')
    assert all(t['score'] == .986 and t['species_confidence'] is None for t in rows)
    assert [t['scientific_name'] for t in rows if t['auto_select']] == (['Vireo swainsoni'] if preferred else [])
    assert original == dict(species_id='10573', confidence=.986)


def test_warbling_vireo_labels_pipeline_and_manual_confirmation(tmp_path):
    from bird_desktop.data.taxonomy_splits import WARBLING_VIREO
    path = tmp_path / 'vireo.csv'
    path.write_text('species_id,scientific_name,name_en\n10573,Vireo gilvus,Eastern Warbling Vireo\n')
    assert splits_for_labels(path) == {'10573': WARBLING_VIREO}
    names = BirdNamesLoader(path)
    assert names.get('10573').name == 'Warbling Vireo (Eastern / Western)'
    assert names.get('taxonomy:Vireo swainsoni').name == 'Western Warbling Vireo'
    pipeline = BirdOnnxPipeline.__new__(BirdOnnxPipeline)
    pipeline.class_ids = ['10573']
    pipeline.taxonomy_splits = splits_for_labels(path)
    pipeline._run_classifier = lambda feeds: (np.array([[.986]]), np.array([[0]]))
    result = pipeline._classifier_results_from_feeds({
        'lat': np.array([40.23]), 'lon': np.array([-111.73]), 'day': np.array([146]),
    })[0][0]
    assert result['taxonomy_refinement']['suggestions'][0]['scientific_name'] == 'Vireo swainsoni'
    for scientific in ('Vireo gilvus', 'Vireo swainsoni'):
        bird = {'classification': [result]}
        store = ManualCorrectionStore.__new__(ManualCorrectionStore)
        store._apply_to_bird(bird, f'taxonomy:{scientific}')
        assert bird['classification'][0]['species_id'] == f'taxonomy:{scientific}'
        assert bird['classification'][0]['manual'] is True
        assert 'taxonomy_refinement' not in bird['classification'][0]
        assert bird['manualCorrection']['originalClassification'] == result


def test_future_warbling_vireo_classifier_bypasses_refinement(tmp_path):
    path = tmp_path / 'future-vireo.csv'
    path.write_text('species_id,scientific_name,name_en\n10573,Vireo gilvus,Eastern Warbling Vireo\n20000,Vireo swainsoni,Western Warbling Vireo\n')
    assert splits_for_labels(path) == {}
    names = BirdNamesLoader(path)
    assert names.get('10573').name == 'Eastern Warbling Vireo'
    assert names.get('20000').name == 'Western Warbling Vireo'
    assert len(names.all()) == 2


@pytest.mark.parametrize('lat,lon,day,expected', [
    (40.23, -111.73, 146, 'Vireo swainsoni'),  # Utah May25
    (38.6, -121.5, 180, 'Vireo swainsoni'),   # California
    (45.5, -122.7, 180, 'Vireo swainsoni'),   # Oregon
    (47.6, -122.3, 180, 'Vireo swainsoni'),   # Washington
    (53.9, -122.7, 180, 'Vireo swainsoni'),   # British Columbia
    (61.2, -149.9, 180, 'Vireo swainsoni'),   # Alaska
    (43, -89, 180, 'Vireo gilvus'),          # Wisconsin
    (42.4, -71.1, 180, 'Vireo gilvus'),      # Massachusetts
    (43.7, -79.4, 180, 'Vireo gilvus'),      # Ontario
    (50.5, -104.6, 180, 'Vireo gilvus'),     # Saskatchewan away from Cypress Hills
    (40, -105.3, 180, None),                # Colorado Front Range
    (43, -106, 180, None),                  # Wyoming
    (42, -103, 180, None),                  # western Nebraska
    (44, -103.5, 180, None),                # Black Hills
    (51, -114, 180, None),                  # Alberta
    (49.7, -109.5, 180, None),              # Cypress Hills
    (47, -110, 180, None),                  # Montana unresolved guidance
    (20, -100, 180, None),                  # central Mexico
    (17, -96, 30, None),                    # Oaxaca winter overlap
    (29, -110.5, 30, 'Vireo swainsoni'),     # Sonora winter supported by Cornell
    (29, -110.5, 280, 'Vireo swainsoni'),    # Sonora autumn
    (40.23, -111.73, 134, None),
    (40.23, -111.73, 135, 'Vireo swainsoni'),
    (40.23, -111.73, 212, 'Vireo swainsoni'),
    (40.23, -111.73, 213, None),
    (43, -89, 146, None),                   # Eastern core spring migration
    (43, -89, 156, None),
    (43, -89, 157, 'Vireo gilvus'),
    (43, -89, 212, 'Vireo gilvus'),
    (43, -89, 213, None),
    (43, -89, 30, None),                    # North American winter vagrant
    (40.23, -111.73, None, None),
    (None, -111.73, 180, None),
])
def test_broader_warbling_regional_and_seasonal_hints(lat, lon, day, expected):
    from bird_desktop.data.taxonomy_splits import WARBLING_VIREO
    result = refine(dict(species_id='10573', confidence=.986), WARBLING_VIREO, lat, lon, day)
    refinement = result['taxonomy_refinement']
    assert [t['scientific_name'] for t in refinement['suggestions'] if t['preferred']] == ([expected] if expected else [])
    assert refinement['requires_confirmation'] is True
    assert result['confidence'] == .986
    assert all(t['confidence'] is None for t in refinement['suggestions'])


def test_reusable_regional_rules_leave_conflicts_and_boundaries_unresolved():
    from bird_desktop.data.geographic_refinement import RegionalRule, regional_preference
    a = RegionalRule('a', 'A', ((0, 10, 0, 10),), (100, 200))
    b = RegionalRule('b', 'B', ((5, 15, 5, 15),), (100, 200))
    assert regional_preference((a, b), 7, 7, 150)[0] is None
    assert regional_preference((a,), 0, 5, 150)[0] is None
    assert regional_preference((a,), 5, 5, 99)[0] is None
    assert regional_preference((a,), 5, 5, 150, ((4, 6, 4, 6),))[0] is None
    assert regional_preference((a,), 5, 5, 150)[0] == 'A'
