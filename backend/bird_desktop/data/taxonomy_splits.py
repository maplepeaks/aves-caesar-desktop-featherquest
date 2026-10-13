"""Conservative suggestions for taxa missing from an exported classifier.

Never redistribute a classifier probability among daughter species. Geographic
suggestions are review aids, not new predictions or occurrence guarantees.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


TAXONOMY_REVISION = "2025-yellow-warbler-v1"


@dataclass(frozen=True)
class Taxon:
    scientific_name: str
    common_name: str

    @property
    def species_id(self) -> str:
        return f"taxonomy:{self.scientific_name}"

    def suggestion(self, preferred: bool) -> dict:
        return dict(species_id=self.species_id, scientific_name=self.scientific_name,
                    common_name=self.common_name, preferred=preferred, confidence=None)


@dataclass(frozen=True)
class Split:
    original_scientific_name: str
    display_name: str
    taxa: tuple[Taxon, ...]


YELLOW_WARBLER = Split("Setophaga petechia", "Yellow Warbler (Northern / Mangrove)", (
    Taxon("Setophaga aestiva", "Northern Yellow Warbler"),
    Taxon("Setophaga petechia", "Mangrove Yellow Warbler"),
))
SPLITS = (YELLOW_WARBLER,)


@lru_cache(maxsize=8)
def splits_for_labels(path: Path) -> dict[str, Split]:
    """Do not override a future model that actually exports both taxa.

    Labels are immutable model artifacts; callers replace them when replacing
    the model. Read by scientific name, never by a guessed class index.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    scientific_names = {row.get("scientific_name", "").strip() for row in records}
    result = {}
    for split in SPLITS:
        if all(taxon.scientific_name in scientific_names for taxon in split.taxa):
            continue
        for index, row in enumerate(records):
            if row.get("scientific_name", "").strip() == split.original_scientific_name:
                result[str(row.get("species_id") or index).strip()] = split
    return result


def _yellow_warbler_preference(latitude, longitude, day):
    try:
        lat, lon = float(latitude), float(longitude)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            return None, "missing_or_invalid_coordinates"
    except (TypeError, ValueError):
        return None, "missing_or_invalid_coordinates"
    # Deliberately coarse review hints, not range polygons. Northern is the
    # expected taxon in continental North America; Mangrove vagrants exist.
    if 32 <= lat <= 72 and -170 <= lon <= -50:
        return "Setophaga aestiva", "continental_north_america; vagrants_possible"
    # Galapagos resident population belongs to Mangrove. No season needed.
    if -2 <= lat <= 1 and -92 <= lon <= -89:
        return "Setophaga petechia", "galapagos_resident; vagrants_possible"
    # Northern migrants overlap resident Mangrove in tropical wintering areas.
    # Only offer a resident preference for Caribbean June/July photographs.
    try:
        summer = math.isfinite(float(day)) and 153 <= float(day) <= 212
    except (TypeError, ValueError):
        summer = False
    if summer and 15 <= lat <= 26 and -85 <= lon <= -60:
        return "Setophaga petechia", "caribbean_summer_resident; vagrants_possible"
    return None, "overlapping_or_unsupported_range_or_season"


def refine(classification: dict, split: Split, latitude=None, longitude=None, day=None) -> dict:
    preferred, reason = (_yellow_warbler_preference(latitude, longitude, day)
                         if split == YELLOW_WARBLER else (None, "no_geographic_rule"))
    suggestions = [taxon.suggestion(taxon.scientific_name == preferred) for taxon in split.taxa]
    suggestions.sort(key=lambda item: not item["preferred"])
    return dict(classification, taxonomy_refinement=dict(
        revision=TAXONOMY_REVISION, status="suggested" if preferred else "unresolved",
        method="geographic_hint" if preferred else "taxonomy_split", reason=reason,
        original_species_id=classification["species_id"],
        original_scientific_name=split.original_scientific_name,
        model_confidence=classification["confidence"], requires_confirmation=True,
        suggestions=suggestions,
    ))


def review_candidates(classification: dict, common_name: str, scientific_name: str) -> list[dict]:
    """Flatten review suggestions for integrations without inventing scores.

    ``score`` remains the original model score (for ranking legacy results).
    Consumers must label it as such when ``taxonomy_refinement`` is present.
    A suggested daughter taxon has no model probability of its own.
    """
    refinement = classification.get("taxonomy_refinement")
    if not refinement:
        return [dict(common_name=common_name, scientific_name=scientific_name,
                     score=classification["confidence"])]
    return [dict(common_name=item["common_name"], scientific_name=item["scientific_name"],
                 score=classification["confidence"], species_confidence=None,
                 auto_select=item["preferred"], taxonomy_refinement=refinement)
            for item in refinement["suggestions"]]
