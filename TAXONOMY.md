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

## Warbling Vireo (2025 split)

The same pinned labels export class `10573`, *Vireo gilvus*, named Eastern
Warbling Vireo, with no *Vireo swainsoni* class. Training on the former combined
Warbling Vireo concept is plausible but undocumented. Treat this as a coverage
limitation, rather than evidence that the classifier distinguishes Eastern from
Western. The affected display is **Warbling Vireo (Eastern / Western)**, with
both daughter species offered for confirmation and no daughter probabilities.
The raw class ID and score stay unchanged. Confirmed manual selections remain
separate from model evidence.

### Geographic data choice

Cornell's [Status and Trends FAQ](https://science.ebird.org/en/status-and-trends/faq)
explicitly says its Eastern/Western Warbling Vireo product was modeled using
both species. That combined layer cannot distinguish the daughters. Access to
Status products also requires agreeing to the data terms and an access key;
see [downloads](https://science.ebird.org/en/status-and-trends/download-data).
Do not treat a renamed combined product as a split-aware range dataset. Revisit
this choice when separately modeled daughter products become available with
appropriate coverage and redistribution terms. No raster/GIS dependency,
network request, or downloaded biological range dataset is added here.

### Conservative regional implementation

`geographic_refinement.py` supplies reusable declarative regional boxes,
seasonal windows, explicit exclusions, input validation, and conflict handling.
These are hand-authored regional CORE hints, not digitized Cornell range maps,
county boundaries, habitat suitability, or complete biological ranges. Exact
box edges deliberately remain unresolved. Unsupported gaps are intentional.
The rules conditionally suggest a daughter of an existing Warbling Vireo
prediction; they cannot establish that a bird occurs at a coordinate.

Cornell's [2025 regional guidance](https://science.ebird.org/en/use-ebird-data/the-ebird-taxonomy/2025-ebird-taxonomy-update)
provides the basis for the taxon choices. Western cores cover the Pacific coast,
Great Basin/interior Southwest, British Columbia and parts of Alaska/Yukon.
Eastern cores cover central/eastern US, southern eastern Canada, and a
Saskatchewan interior core. Contact-region masks leave Alberta, Cypress Hills,
western Plains/Black Hills, Wyoming and Colorado cores unresolved; Montana is
also deferred. No county-level or habitat-specific assignments are approximated
within these masks, even where some locations could be assigned with finer data.
No continent-wide longitude dividing line is used.

Season windows are deliberately conservative IMPLEMENTATION choices, not
Cornell-published arrival/departure dates: day 135–212 for Pacific/intermountain
cores (approximately mid-May–July); day 157–212 for eastern/northern cores
(approximately June 5/6–July). The classifier supplies day-of-year without year,
so endpoints shift a day in leap years. Spring/fall North American migration
outside those windows and winter remain unresolved. Small interior cores of
Sonora, Sinaloa and Nayarit support Western during days 1–120 and 244–366,
following Cornell's explicit western-Mexico migration/winter guidance; other
Mexican regions remain unresolved. This does not implement the full Mexican
resident/breeding range. Missing/invalid GPS or date always stays unresolved.

Vagrants remain possible in every supported region. All hints require human
confirmation and never acquire a numeric daughter-species confidence.

Refinement alternatives belong to the same detected bird. PhotoCatalog keeps
one detection and excludes taxonomy alternatives from independent species
discovery. A future labels file containing both daughter species bypasses the
legacy rule. Changing the taxonomy revision invalidates cached predictions.
