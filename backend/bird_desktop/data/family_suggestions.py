"""Family review evidence from exported species candidates, not probabilities.

The current ONNX export returns only top five species. The pinned graph applies Softmax before TopK, but these five scores
exclude the remaining probability mass. Never normalize this incomplete evidence or describe its
sum as family confidence. Group original classes BEFORE split alternatives.
Taxonomy mappings are supplied by the integration, not guessed from names.
"""
import math

FAMILY_EVIDENCE_REVISION = 'top-candidates-family-v1'


def family_suggestions(classifications, class_families):
    grouped = {}
    seen = set()
    for item in classifications:
        identity = item['species_id']
        if identity in seen:
            continue
        seen.add(identity)
        family = class_families.get(identity)
        score = item.get('confidence')
        if not family or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            continue
        key = family['scientific_name']
        row = grouped.setdefault(key, dict(
            family, category='family', evidence_method='partial_top_candidates_sum',
            aggregate_score=0., best_species_score=0., supporting_classes=[],
            family_confidence=None, requires_confirmation=True,
        ))
        row['aggregate_score'] += score
        row['best_species_score'] = max(row['best_species_score'], score)
        row['supporting_classes'].append(identity)
    return sorted(grouped.values(), key=lambda row: row['aggregate_score'], reverse=True)
