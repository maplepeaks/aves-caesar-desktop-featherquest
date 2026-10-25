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

from .geographic_refinement import RegionalRule, regional_preference


TAXONOMY_REVISION = "2025-splits-v4"


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
WARBLING_VIREO = Split("Vireo gilvus", "Warbling Vireo (Eastern / Western)", (
    Taxon("Vireo gilvus", "Eastern Warbling Vireo"),
    Taxon("Vireo swainsoni", "Western Warbling Vireo"),
))
WHIMBREL = Split("Numenius phaeopus", "Whimbrel (Hudsonian / Eurasian)", (
    Taxon("Numenius hudsonicus", "Hudsonian Whimbrel"),
    Taxon("Numenius phaeopus", "Eurasian Whimbrel"),
))
SPLITS = (YELLOW_WARBLER, WARBLING_VIREO, WHIMBREL)


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


# Regional cores derived conservatively from Cornell's 2025 taxonomy guidance.
# These are NOT digitized ranges or county boundaries. Leave boundary areas out.
# Pacific/Intermountain cores can support late-May hints; near the Plains use
# only the narrower June/July breeding window (day 157 starts June 5/6).
WARBLING_REGIONS = (
    RegionalRule("western_pacific_intermountain_breeding", "Vireo swainsoni", (
        (33, 42, -123, -114),       # California/interior Southwest
        (42, 49, -124, -117),       # Pacific Northwest
        (36, 45, -117, -109.1),     # Great Basin/Utah/Idaho, away from contact
        (49, 59, -130, -120.1),     # British Columbia core, excluding Alberta
    ), (135, 212)),
    RegionalRule("western_northern_breeding", "Vireo swainsoni", (
        (59, 64, -150, -125),       # Southwestern Alaska/southern Yukon cores
    ), (157, 212)),
    RegionalRule("eastern_breeding_core", "Vireo gilvus", (
        (36, 49, -96, -76),         # Central/eastern US, away from Plains contact
        (40, 47, -76, -69),         # Northeast
        (45, 50, -79, -64),         # Southern Ontario/Quebec/Maritimes cores
        (49, 53, -105, -98),        # Saskatchewan core, north/east of Cypress Hills
    ), (157, 212)),
    # Cornell explicitly supports Western in these western Mexican regions in
    # migration/winter. Restrict to interior cores of Sonora/Sinaloa/Nayarit;
    # no inference for southern/eastern Mexico or central-American overlap.
    RegionalRule("western_mexico_nonbreeding_core", "Vireo swainsoni", (
        (28, 30, -111.5, -110),     # Sonora
        (24, 26, -107.5, -106.5),   # Sinaloa
        (21.5, 22, -105.1, -104.8), # Nayarit
    ), (1, 120)),
    RegionalRule("western_mexico_autumn_core", "Vireo swainsoni", (
        (28, 30, -111.5, -110), (24, 26, -107.5, -106.5),
        (21.5, 22, -105.1, -104.8),
    ), (244, 366)),
)
# Deliberately exclude entire contact-region cores instead of pretending GPS
# alone resolves county/habitat-specific differences described by Cornell.
WARBLING_CONTACT = (
    (37, 41, -109.1, -102),  # Colorado
    (41, 45, -111.1, -104),  # Wyoming
    (40, 43, -104.1, -98),   # western Nebraska/Plains
    (43, 46, -104.1, -100),  # Black Hills/western South Dakota
    (49, 60, -120, -110),    # Alberta
    (49, 50.5, -110, -108),  # Cypress Hills/Maple Creek
    (45, 49, -117, -104),    # Montana: Cornell conversion still in progress
)


def _warbling_vireo_preference(latitude, longitude, day):
    return regional_preference(WARBLING_REGIONS, latitude, longitude, day,
                               WARBLING_CONTACT)


# Continental review cores, not occurrence maps. Exclude Beringia, Greenland,
# Iceland and remote oceanic islands rather than infer across overlap/vagrancy.
# No seasonal distinction is needed for these conditional continental hints.
WHIMBREL_REGIONS = (
    RegionalRule("americas_core", "Numenius hudsonicus", (
        (7, 50, -130, -50),
        (50, 70, -160, -50),
        (-56, 13, -82, -35),
    ), (1, 366)),
    RegionalRule("old_world_core", "Numenius phaeopus", (
        (35, 70, -10, 160),
        (5, 35, -18, 50),
        (-35, 5, 8, 50),
        (-10, 35, 50, 150),
        (-40, -10, 110, 155),
    ), (1, 366)),
)


def _whimbrel_preference(latitude, longitude, day):
    # Reuse coordinate/conflict/edge validation without requiring a photo date:
    # these rules express geography only, not seasonal occurrence likelihood.
    return regional_preference(WHIMBREL_REGIONS, latitude, longitude, 1)


def refine(classification: dict, split: Split, latitude=None, longitude=None, day=None) -> dict:
    rules = {YELLOW_WARBLER: _yellow_warbler_preference,
             WARBLING_VIREO: _warbling_vireo_preference,
             WHIMBREL: _whimbrel_preference}
    rule = rules.get(split)
    preferred, reason = (rule(latitude, longitude, day) if rule
                         else (None, "no_geographic_rule"))
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
