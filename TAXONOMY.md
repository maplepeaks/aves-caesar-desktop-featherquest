# Taxonomy splits and model coverage

The pinned classifier release (`f9b2159306029fbaa8f6c9a7799776b422278ea2`)
exports 10,859 classes. Its label table includes `9152`, *Setophaga petechia*,
named Mangrove Yellow Warbler, but no *Setophaga aestiva* class. The 2025
eBird/Clements checklist split Northern Yellow Warbler (*S. aestiva*) from the
former Yellow Warbler (*S. petechia*).

The model cannot select Northern under these labels. Its training taxonomy is
not documented in the model card, so the available artifacts do not establish
whether this class was trained on the former combined species or only Mangrove.
This is a demonstrated taxonomy coverage problem; it does not prove that every
prediction of class 9152 is a visual identification error.

## Conservative correction

`data/taxonomy_splits.py` declares affected splits by scientific name and enables
them only when daughter taxa are missing from the labels. Class indices, ONNX
weights and exported CSV files stay unchanged. The affected class is displayed
as **Yellow Warbler (Northern / Mangrove)**. Both species remain available as
manual corrections with separate namespaced identifiers.

Every affected prediction retains its original `species_id` and `confidence`.
Its `taxonomy_refinement` contains the original scientific name, model score,
two suggestions with **null species confidence**, the rule version, reason and
an explicit confirmation requirement. No numerical daughter-species confidence
is inferred. Cached predictions are invalidated when these rules change.

GPS and day of year already enter the ONNX classifier as `lat`, `lon`, and `day`;
absent values are NaN. There is no separate authoritative distribution database
in this app. These inputs cannot recover a species absent from its output labels.

For review only, coarse geographic hints prefer Northern in the North American
box 32–72° N, 170–50° W, and Mangrove around the Galapagos (2° S–1° N,
92–89° W). A Caribbean summer hint (15–26° N, 85–60° W, day 153–212) prefers
resident Mangrove. These boxes are deliberately limited **heuristics**, not
range polygons or exclusion rules. Vagrants and exceptional summering birds
are possible. A date is not sufficient by itself. Mainland tropical areas,
Caribbean winter or unknown season, unsupported regions and missing/invalid
coordinates remain unresolved. Northern migrants overlap Mangrove in tropical
wintering areas; geography cannot reliably separate every photograph.

Additional splits can reuse the declaration and result structure, but should
not inherit Yellow Warbler's geographic rules. A future classifier exporting
both daughter taxa bypasses this correction.

`review_candidates()` is an optional integration helper. It exposes both names
and preserves the original model score for ranking; clients must not label that
score as a daughter-species probability. Only a geographically preferred
suggestion has `auto_select=true`; all still require human confirmation.

## Sources and validation

- [Cornell 2025 changes](https://www.birds.cornell.edu/clementschecklist/updates-and-corrections-october-2025/)
- [eBird 2025 taxonomy update](https://science.ebird.org/en/use-ebird-data/the-ebird-taxonomy/2025-ebird-taxonomy-update)
- [Northern Yellow Warbler range](https://www.allaboutbirds.org/guide/Yellow_Warbler/maps-range)
- [Mangrove Yellow Warbler range](https://www.allaboutbirds.org/guide/Mangrove_Yellow_Warbler/maps-range)
- [Pinned model artifacts](https://huggingface.co/avescaesar/bird-detect-classify/tree/f9b2159306029fbaa8f6c9a7799776b422278ea2)

Run `PYTHONPATH=backend python -m pytest tests/test_taxonomy_splits.py`.
Tests cover Havasupai, Galapagos, Caribbean summer/winter, missing dates/GPS,
invalid coordinates, future split-aware labels, per-image batch context,
unchanged raw scores and indices, integration output, and cache invalidation.
