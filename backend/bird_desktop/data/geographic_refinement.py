"""Dependency-free regional review hints, not biological range polygons.

Boxes deliberately select regional cores. Uncovered locations and conflicting
rules stay unresolved. Seasons use the classifier's day-of-year input; no year
is available, so endpoints are deliberately conservative around leap years.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RegionalRule:
    name: str
    taxon: str
    boxes: tuple[tuple[float, float, float, float], ...]  # south, north, west, east
    season: tuple[int, int]

    def matches(self, latitude, longitude, day):
        return self.season[0] <= day <= self.season[1] and any(
            south < latitude < north and west < longitude < east
            for south, north, west, east in self.boxes)


def regional_preference(rules, latitude, longitude, day, exclusions=()):
    try:
        lat, lon, ordinal = float(latitude), float(longitude), float(day)
        valid = (all(math.isfinite(v) for v in (lat, lon, ordinal))
                 and -90 <= lat <= 90 and -180 <= lon <= 180
                 and 1 <= ordinal <= 366 and ordinal.is_integer())
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return None, 'missing_or_invalid_coordinates_or_season'
    if any(s <= lat <= n and w <= lon <= e for s, n, w, e in exclusions):
        return None, 'contact_or_overlap_region'
    matches = [rule for rule in rules if rule.matches(lat, lon, ordinal)]
    if len({rule.taxon for rule in matches}) == 1:
        return matches[0].taxon, matches[0].name + '; vagrants_possible'
    return None, 'overlapping_or_unsupported_range_or_season'
