# LLM-as-Judge Calibration & Generator Feedback Loop

This project is a small, self-contained demonstration of LLM-as-judge infrastructure: building a judge that can be validated against known-correct answers, and then closing the loop by using that judge to give a generator feedback and measuring whether it improves. It uses the `vicgalle/creative-rubrics` dataset, filtered to a single creative-writing style ("Chaos Cinema Critique — A Manifesto for the Absurd"), as a stand-in domain. The goal of this project is to prove out the infrastructure and methodology — golden-set validation, calibration loops, batched/cached API usage, generator-judge feedback — not to produce a polished creative-writing model. Where that distinction matters, it's called out below.

## How It Fits Together

The project has two phases that build on each other. **Phase 1** builds and validates a judge: can an LLM reliably grade whether a piece of writing matches a target style, checked against ground-truth tier labels the dataset already provides? **Phase 2** only runs if Phase 1 succeeds — it uses the validated judge to grade a generator's attempts at writing in the style, feeds the generator per-item feedback, and tracks whether its scores improve over a few rounds.

Everything is orchestrated by `harness.py`, which calls into `data_prep.py` and `generator_prompts.py` to prepare data, then into `judge.py` and `generator.py` to do the actual LLM work. The flow below follows that same order: data first, then the judge, then the generator, then how the harness ties them together.

## Preparing the Data (`data_prep.py`, `generator_prompts.py`)

`data_prep.py` loads the dataset from Hugging Face and filters it to a single style with full coverage across all three ground-truth tiers (GOOD / OK / BAD). It then splits the filtered rows into three disjoint sets for the judge:

- **Few-shot set** (8 rows) — shown to the judge once as calibration examples, never scored.
- **Calibration set** (40 rows) — what the judge is repeatedly graded against, round after round, with feedback in between.
- **Golden set** (12 rows) — held out completely until calibration is done. Touched exactly once, as a final honesty check.

`generator_prompts.py` builds a second, separate set of splits for the generator — few-shot examples to prime its writing, a fixed set of prompts it's tracked against every round, and a small held-out set for a one-time generalization check at the end. These are pulled from the judge's calibration/golden rows (never from the judge's own few-shot set, so the judge isn't grading topics it already has a labeled example for), and the generator's few-shot rows are kept separate from its own tracked/held-out prompts for the same reason. Before anything runs, the harness prints a table of exactly which row IDs went into which bucket, so overlaps are visible before any API calls are spent.

## The Judge (`judge.py`)

The `Judge` class wraps a single job: given a batch of prompt/response pairs, grade each one GOOD / OK / BAD against the style's rubric, with brief reasoning per item. A few things about how it's built:

- **One call per round, not one per item.** `grade_batch()` sends an entire round's worth of items in a single API call and asks for a JSON array back, rather than looping and paying for the full rubric + few-shot text on every single item.
- **Prompt caching.** The static portion of the system prompt (rubric + few-shot examples) is marked cacheable and kept separate from the feedback block, which changes every round. This means the expensive, unchanging part of the prompt is billed once and reused cheaply across the whole run instead of being repaid every call.
- **Feedback loop.** After each round, the harness builds a plain-language summary of the judge's confusion matrix ("you graded N GOOD items as OK...") and feeds it back for the next round.
- Runs on `claude-haiku-4-5-20251001` — classification-style grading doesn't need a larger model, and Haiku is both cheaper and, as it turned out, easier to keep within a predictable token budget.

## The Generator (`generator.py`)

The `Generator` class is structurally the same pattern as the judge — cached static system prompt, batched calls, feedback kept separate — but its job is the reverse: given a batch of topics, write a review in the target style. Feedback into the generator is **per-item**, not aggregate, since aggregate feedback alone showed essentially no improvement in the judge's own calibration loop (see Findings below) — the working assumption was that per-item detail ("this one was weak because X") would give the generator something more actionable to work with.

Two bugs surfaced and got fixed here that are worth knowing about if you're reading the code:
- Claude Sonnet 5 runs adaptive thinking **on by default**, even with no `thinking` parameter passed — unlike earlier models, where omitting it meant no thinking ran. Thinking tokens count against `max_tokens`, so this silently ate into the budget meant for review text and caused truncated, unparseable responses. Fixed by explicitly passing `thinking={"type": "disabled"}`.
- Even with thinking off, the original per-item token budget still undershot how verbose the model got with this style. Fixed by raising the budget substantially and adding explicit length guidance to the system prompt (roughly 150–300 words per review), so the model isn't relying on an ever-larger token ceiling to avoid getting cut off.

The generator now defaults to `claude-haiku-4-5-20251001` as well (originally built on Sonnet 5) — cheaper, and it predates the tokenizer change that made Sonnet 5 responses cost more tokens for the same text.

## The Harness (`harness.py`)

`harness.py` runs both phases in sequence.

**Phase 1** runs the judge calibration loop (up to 5 rounds, default 90% exact-match threshold, reshuffling the calibration set every round to avoid positional memorization), then the one-shot golden-set check, and writes a timestamped report.

**Phase 2 is golden-gated** — it only runs if the judge's golden-set performance clears a trigger threshold (default 80%), so the generator never gets handed a judge that hasn't been validated. If triggered, it runs the generator through its fixed round-tracking prompts for up to 3 rounds (writing generated reviews to a readable file each round for manual spot-checking — this matters, see Findings), collecting per-round average judge scores, then does a final one-shot check against held-out prompts to test whether the generator generalized beyond the specific prompts it practiced on.

## Setup

```bash
pip install datasets anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
python harness.py
```

First run downloads the filtered dataset from Hugging Face (a few MB). A full run — judge calibration plus the generator phase — makes on the order of 10-15 API calls total, thanks to batching.

## Example Output

```
================================================================================
JUDGE CALIBRATION REPORT
================================================================================
Style graded: Chaos Cinema Critique - A Manifesto for the Absurd
Calibration set size: 40 | Golden set size: 12
Threshold: 90.0% exact match rate

Round-by-round performance:
  Round 1: exact_match=0.68, mean_off_by=0.35, cumulative_off_by=14
  Round 2: exact_match=0.65, mean_off_by=0.38, cumulative_off_by=15
  Round 3: exact_match=0.70, mean_off_by=0.33, cumulative_off_by=13
  ...

Golden set (held-out, one-shot):
  exact_match=0.92, mean_off_by=0.08

VERDICT: Judge is USABLE for downstream generator feedback.

================================================================================
PHASE 2: GENERATOR TRIGGER CHECK
================================================================================
✓ Golden set performance (92%) cleared trigger threshold (80%). Proceeding...

Generator improvement summary:
  Round 1: avg score = 1.50
  Round 2: avg score = 1.38
  Round 3: avg score = 1.62
  Held-out: avg score = 1.67
```

## Key Design Decisions

- **Golden-set discipline everywhere.** Both the judge and the generator hold out a set that's touched exactly once, after their respective loops finish. This is the one honest signal in the project that isn't subject to feedback-loop overfitting.
- **Batching + prompt caching over per-item calls.** The first working version of this project called the API once per graded item; a full run cost several dollars. Restructuring to one call per round, with the static rubric/few-shot content cached, cut this dramatically.
- **Confusion matrix, not just a scalar score.** Both judge and generator scoring track the full confusion matrix, not just an aggregate off-by metric, so directional bias is visible if it exists.
- **Per-item feedback for the generator, aggregate for the judge** — a deliberate, tested difference (see Findings) rather than an oversight.

## Findings

A few things came out of running this that are worth stating plainly rather than glossing over:

- **The judge's own calibration curve was flat across rounds.** Feedback there is only an aggregate confusion-matrix summary ("you graded N GOOD as OK"), which gives the model no example-level detail to act on beyond what its few-shot examples already provided. This isn't a bug — it's a real, informative result about the limits of aggregate-only feedback, and it's part of why the generator was built to receive per-item feedback instead.
- **The generator's score did improve modestly (roughly 1.50 → 1.62 across valid rounds), but a manual read of the generated text complicates a clean "it got better" story.** The writing was strong from round 1 and stayed strong, but it converged on and reused the same handful of stylistic devices (hex-color-coded abstractions, pseudocode fragments, "you become the X" perspective inversions, a closing italicized rhetorical question) across nearly every topic, regardless of subject matter. That looks less like the generator learning nuanced, topic-adaptive style and more like it finding a high-scoring template early and reliably re-executing it.
- **Per-item feedback structurally can't catch that kind of pattern.** The judge grades each generated review independently against the rubric, so its per-item reasoning only ever addresses "does this one review match the style" — it has no mechanism to notice or flag "you used this same device in 6 of the last 8 reviews." This is a real, worth-knowing limitation of per-item feedback as a mechanism, not a wiring bug (confirmed directly — the feedback pipeline was checked and is delivering real, judge-derived content to the generator every round).

## Next Steps

- **Richer, multi-dimensional rubric.** The current setup uses a single 3-tier ordinal grade (GOOD/OK/BAD). A judge that scores multiple independent dimensions (e.g., voice, structural variety, topical relevance) would both give more diagnostic signal and make the trope-convergence problem visible in the metrics themselves, rather than only via manual review.
- **Add batch-level feedback alongside per-item.** Since per-item feedback can't see across items, a lightweight batch-level pass — even just asking the judge "what stylistic device did you see repeated across this batch?" — could directly target the convergence problem found above.
- **Increase token budget / relax the length cap for longer-form output.** The current length guidance (150–300 words) was added to control cost and truncation risk; loosening it would be a reasonable next experiment now that the truncation root cause (adaptive thinking silently consuming the output budget) is understood and fixed.
- **Pairwise judging as an alternative to absolute scoring.** This project uses pointwise (absolute 1-of-3-tiers) grading, chosen deliberately for per-dimension diagnostic signal. Pairwise comparison ("which of these two is closer to the style") is generally considered more reliable per-judgment in current practice and would be a reasonable comparison point.
- **Source-blinding, if human-written examples are ever mixed in.** The current dataset is entirely LLM-generated, so this isn't yet a live risk — but if real human-written examples were added to the judge's calibration set, source metadata should be stripped so the judge can't shortcut on "does this look machine-written" rather than actually grading style match.

## Scope

**Included:** LLM judge calibration and validation against ground truth, prompt-based grading (no fine-tuning), a full generator feedback loop gated on judge validation, batched + cached API usage, confusion-matrix-level metrics and reporting.

**Explicitly out of scope:** fine-tuning either the judge or the generator, automated validation of the judge's reasoning quality (the dataset doesn't provide reasoning ground truth to check against), and any use of the generator's output to actually update the generator's weights (the feedback loop is prompt-based iterative refinement, not RLAIF).

## Files

- `data_prep.py` — dataset loading, filtering, and judge-side splitting
- `generator_prompts.py` — generator-side prompt/few-shot splitting, with overlap checks against the judge's splits
- `judge.py` — LLM judge wrapper (batched, cached, aggregate feedback)
- `generator.py` — LLM generator wrapper (batched, cached, per-item feedback)
- `harness.py` — orchestrates both phases, computes metrics, writes reports
- `judge_report_*.md` / `*.json` — timestamped Phase 1 reports
- `generations_round_*.md` / `*.json` — timestamped generator output, saved every round for manual review