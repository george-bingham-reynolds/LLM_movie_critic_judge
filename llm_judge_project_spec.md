# LLM-as-Judge Weekend Project — Spec for Implementation

## Goal
Build and validate an LLM judge that scores how well a movie review matches a target rubric/style, using the `vicgalle/creative-rubrics` dataset. Judge-only scope (no generator yet). Deliverable: a callable script (the "harness") that runs the full calibration loop and reports whether the judge is trustworthy.

---

## 0. Data prep

- Load `vicgalle/creative-rubrics` from Hugging Face.
- **Filter to a single rubric/style** (pick one movie-review style consistently present across all three score tiers — highest / 50-100 / lowest — so you have full coverage within that one style).
- Each row after filtering should give you: `prompt`, `response_text`, `tier` (high / mid / low).
- **Split into three disjoint sets** (no overlap):
  - `few_shot_set` — small (~5-8 examples), one from each tier ideally. Shown to the judge once, up front, never scored.
  - `calibration_set` — larger (~30-50 examples). The judge scores this repeatedly across rounds, gets feedback, retries.
  - `golden_set` — small (~10-15 examples), held out completely until the very last step. One-shot final check, never seen before that point.
- Shuffle order within calibration_set on every round (avoid positional memorization).

```python
# pseudocode
dataset = load_dataset("vicgalle/creative-rubrics")
filtered = filter_to_single_style(dataset, style="<pick one>")
few_shot_set, calibration_set, golden_set = split(filtered, sizes=[8, 40, 12], stratify_by="tier")
```

---

## 1. Judge system prompt (few-shot primer)

Sent once, at the start of every run. Includes the rubric and few-shot examples.

```
SYSTEM PROMPT:

You are grading how well a written movie review matches a specific target style.
Style being graded: <STYLE NAME>
Rubric for this style: <RUBRIC TEXT FROM DATASET>

You will be shown a prompt and a response. Grade the response as one of:
- GOOD  (fully matches the style/rubric)
- OK    (partially matches — some elements present, others missing or off)
- BAD   (does not match the style/rubric)

Before giving your grade, briefly reason through how well the response satisfies
the rubric (2-3 sentences). Then output your final grade on its own line as:
GRADE: <GOOD|OK|BAD>

Here are labeled examples to calibrate your grading:

Example 1:
Prompt: <few_shot prompt 1>
Response: <few_shot response 1>
Reasoning: <you can write a short rubric-based rationale here, or leave brief since ground truth is tier-only>
GRADE: <tier 1>

[... repeat for all few-shot examples ...]

Now you will be shown new prompt/response pairs one at a time (or in a batch — see harness note below). Apply the same standard.
```

**Note on few-shot "Reasoning" lines:** the dataset only has final tiers, not example reasoning — so keep the reasoning field in your few-shot examples short and generic ("meets tone and structure expected of this style" / "misses key stylistic markers") rather than inventing detailed justifications you can't verify. Don't over-engineer this part.

---

## 2. Calibration loop

```python
# pseudocode

def score_round(judge, calibration_set, round_num):
    shuffled = shuffle(calibration_set)
    predictions = []
    for item in shuffled:
        judge_output = judge.grade(prompt=item.prompt, response=item.response_text)
        # judge_output = {"reasoning": "...", "grade": "GOOD"|"OK"|"BAD"}
        predictions.append({
            "id": item.id,
            "true_tier": item.tier,
            "pred_tier": judge_output["grade"],
            "reasoning": judge_output["reasoning"],
        })
    return predictions

def score_metrics(predictions):
    tier_to_num = {"BAD": 0, "OK": 1, "GOOD": 2}
    off_by_total = 0
    confusion = defaultdict(int)  # (true, pred) -> count
    for p in predictions:
        true_n = tier_to_num[p["true_tier"]]
        pred_n = tier_to_num[p["pred_tier"]]
        off_by_total += abs(true_n - pred_n)
        confusion[(p["true_tier"], p["pred_tier"])] += 1
    return {
        "cumulative_off_by": off_by_total,
        "mean_off_by": off_by_total / len(predictions),
        "confusion_matrix": dict(confusion),
        "exact_match_rate": sum(1 for p in predictions if p["true_tier"] == p["pred_tier"]) / len(predictions),
    }

def build_feedback_message(metrics):
    # summarize confusion matrix into plain language for the judge to read
    # e.g. "You graded 6 GOOD items as OK, 2 OK items as BAD, 0 BAD items as GOOD..."
    return format_confusion_as_feedback(metrics["confusion_matrix"])

# main calibration loop
MAX_ROUNDS = 5
round_history = []

for round_num in range(1, MAX_ROUNDS + 1):
    predictions = score_round(judge, calibration_set, round_num)
    metrics = score_metrics(predictions)
    round_history.append({"round": round_num, **metrics})

    if metrics["exact_match_rate"] >= ACCEPTABLE_THRESHOLD:  # e.g. 0.9, pick a number and justify it
        break

    feedback = build_feedback_message(metrics)
    judge.receive_feedback(feedback)  # append to judge's running context for next round
```

**On the threshold:** pick a number you can defend (e.g. "90% exact match, since with 3 tiers a random baseline is ~33%, so 90% is a meaningful signal over chance") rather than an arbitrary round-trip. Say this explicitly in your writeup — it shows you thought about what "good enough" means rather than picking a number that felt right.

---

## 3. Golden set — final check

Only run this once, after the calibration loop exits (whether by hitting threshold or by hitting MAX_ROUNDS).

```python
golden_predictions = score_round(judge, golden_set, round_num="golden")
golden_metrics = score_metrics(golden_predictions)
```

If golden performance is meaningfully worse than final calibration performance, that's a real and worth-reporting finding — it suggests some overfitting to the calibration set's specific examples/feedback pattern, even without gradient-based training. Say so if it happens; it's a legitimate, interesting result, not a failure of the project.

---

## 4. Harness output

The script should print/save a report like:

```
=== JUDGE CALIBRATION REPORT ===

Style graded: <style>
Calibration set size: N | Golden set size: N

Round-by-round performance:
Round 1: exact_match=0.55, mean_off_by=0.60, cumulative_off_by=24
Round 2: exact_match=0.68, mean_off_by=0.38, cumulative_off_by=15
Round 3: exact_match=0.78, mean_off_by=0.25, cumulative_off_by=10  <- threshold hit

Confusion matrix (final calibration round):
              pred:BAD  pred:OK  pred:GOOD
true:BAD         12        2         0
true:OK           1       15         2
true:GOOD         0        1        17

Golden set (held-out, one-shot):
exact_match=0.75, mean_off_by=0.25

VERDICT: Judge improved 23 percentage points over 3 rounds (0.55 -> 0.78) on
calibration set, and held at 0.75 on unseen golden set. Judge is usable for
downstream generator feedback.
```

Save this to a timestamped file (`judge_report_<timestamp>.json` or `.md`) each run, so you have a record across iterations if you tweak the prompt later.

---

## 5. What NOT to build (scope guard for a weekend)

- No generator yet — that's a separate follow-on step, deliberately out of scope here.
- No fine-tuning — this is a prompted judge, not a trained one.
- No formal validation of the judge's *reasoning* quality — only the final grade is validated against ground truth. Spot-check reasoning by hand on a few examples if curious, but don't build automated reasoning validation; the dataset doesn't support it.
