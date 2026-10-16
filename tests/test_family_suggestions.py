from bird_desktop.data.family_suggestions import family_suggestions
from bird_desktop.data.taxonomy_splits import WARBLING_VIREO, refine

FAMILIES={
    'a':dict(id='family:Anatidae',common_name='Ducks, Geese, and Waterfowl',scientific_name='Anatidae'),
    'b':dict(id='family:Anatidae',common_name='Ducks, Geese, and Waterfowl',scientific_name='Anatidae'),
    'c':dict(id='family:Strigidae',common_name='Owls',scientific_name='Strigidae'),
    '10573':dict(id='family:Vireonidae',common_name='Vireos',scientific_name='Vireonidae'),
}

def test_grouping_partial_mass_without_renormalizing_or_inventing_confidence():
    original=[dict(species_id='a',confidence=.35),dict(species_id='b',confidence=.3),dict(species_id='c',confidence=.25)]
    rows=family_suggestions(original,FAMILIES)
    assert [r['id'] for r in rows]==['family:Anatidae','family:Strigidae']
    assert abs(rows[0]['aggregate_score']-.65)<1e-12
    assert rows[0]['best_species_score']==.35
    assert rows[0]['supporting_classes']==['a','b']
    assert all(r['family_confidence'] is None and r['requires_confirmation'] for r in rows)
    assert all('taxonomy_refinement' not in r for r in rows)
    assert original[0]==dict(species_id='a',confidence=.35)

def test_split_alternatives_and_duplicate_classes_do_not_multiply_evidence():
    original=refine(dict(species_id='10573',confidence=.986),WARBLING_VIREO,40,-111,146)
    rows=family_suggestions([original,original],FAMILIES)
    assert len(rows)==1
    assert rows[0]['aggregate_score']==.986
    assert rows[0]['supporting_classes']==['10573']

def test_unmapped_missing_and_invalid_scores_are_not_guessed():
    rows=family_suggestions([dict(species_id='unknown',confidence=.9),dict(species_id='a',confidence=float('nan')),dict(species_id='b',confidence=-1)],FAMILIES)
    assert rows==[]
    assert family_suggestions([],FAMILIES)==[]
