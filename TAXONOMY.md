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

## Family model evidence

The upstream README reports internal `family_top1_percent` of 99.33 with GPS
and 99.04 without GPS. The shipped `model_performance.json` contains those
constants; runtime model metadata and release-manifest code load/display them.
The public desktop source does not include the evaluation implementation,
held-out labels, confusion matrix or family aggregation procedure, so these
numbers cannot be independently reproduced here. In particular we cannot say
whether family top-1 means mapping the top species to its family or selecting
an aggregated family score. They are population evaluation metrics, not the
confidence of a particular photograph or our new suggestions.

Inspection of the pinned classifier with ONNX Runtime shows exactly two
outputs: `species_top5_scores` and `species_top5_indices`, each batch × 5.
Inspection of the ONNX graph shows Softmax followed by TopK. The full species
vector exists internally (`/Softmax_output_0`) but is not an exported output;
using it would require a modified ONNX export/output graph. We have not changed
weights, exported outputs, detector/classifier inference or dependencies.

`data/family_suggestions.py` groups original returned classes through a supplied
canonical taxonomy mapping. Its sum is **partial top-five score mass**, without
renormalization, calibration or an invented family confidence. Family ranking
can differ from a ranking based on the full vector. Unknown taxonomy mappings
are omitted. Split alternatives are never summed: the original class is counted
once before expansion. Duplicate class IDs are ignored. Every suggestion
requires explicit human confirmation. The helper is reusable by integrations;
the PhotoCatalog adapter supplies authoritative eBird family relationships.


## Whimbrel (2025 split)

The pinned 10,859-class label table has exactly one Whimbrel output:
zero-based index / species ID `6584`, *Numenius phaeopus*, already named
Eurasian Whimbrel. There is no *Numenius hudsonicus* output. As with the previous
splits, the artifacts do not establish whether training used the old combined
species concept; a modern display label alone cannot demonstrate visual
separation of the daughters.

The existing split declaration, name loader, per-image pipeline refinement and
review-candidate helper now expose **Whimbrel (Hudsonian / Eurasian)** with
Hudsonian Whimbrel (*N. hudsonicus*) and Eurasian Whimbrel (*N. phaeopus*)
suggestions. The original classifier ID, score and ordering remain intact.
Suggestions have null species confidence and require confirmation. Revision
`2025-splits-v4` invalidates cached results. A future label table containing
both daughter species bypasses refinement and keeps both native mappings.
No label rows, output indices, model files or weights are changed.

The [eBird 2025 guidance](https://science.ebird.org/en/use-ebird-data/the-ebird-taxonomy/2025-ebird-taxonomy-update)
supports continental geographic inference, with explicit cautions for western
Alaska, mid-ocean islands, Hawaii and Midway. Our hand-authored boxes are a
conservative subset of that guidance, not range polygons or visual evidence.
Americas cores prefer Hudsonian; Europe, Africa, Asia and Australia cores prefer
Eurasian. Western Alaska/Aleutians, Hawaii/Midway, mid-Atlantic islands,
Greenland, Iceland, unsupported gaps and exact outer edges remain unresolved.
These rules deliberately omit some assignable regions rather than implement a
complete global distribution. No date is required: the hint is conditional on
an existing Whimbrel prediction and describes continental affinity, not seasonal
presence. Missing, partial, nonfinite or out-of-bounds GPS yields both alternatives
without a preference. Vagrants remain possible even in supported cores.

Validation in `tests/test_taxonomy_splits.py` covers representative Americas and
Old World locations, ambiguous islands/Beringia, invalid/missing GPS, absent
dates, preserved raw scores, per-image batch context, future split-aware labels,
and all unaffected shipped species names. The shipped CSV is checked for 10,859
rows and Whimbrel at index 6584; pipeline tests preserve the full class-ID order
and verify neighboring class outputs. Existing cache-revision and previous split
tests also remain applicable.
