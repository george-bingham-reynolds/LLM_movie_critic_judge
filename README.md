# LLM Judge Calibration Project

A calibration harness for an LLM-as-judge system that evaluates movie reviews against a specific rubric style using the `vicgalle/creative-rubrics` dataset.

## Overview

This project implements a three-component system:
1. **data_prep.py** - Loads and prepares the dataset
2. **judge.py** - LLM judge wrapper with feedback context
3. **harness.py** - Main calibration loop with metrics and reporting

## Setup

### 1. Install Dependencies

```bash
pip install datasets anthropic
```

### 2. Set API Key

The judge uses Anthropic's Claude API. Set your API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Run the Calibration

```bash
python harness.py
```

## Project Structure

### Data Preparation (`data_prep.py`)

- Loads `vicgalle/creative-rubrics` from HuggingFace
- Filters to **"Chaos Cinema Critique - A Manifesto for the Absurd"** style
- Splits into three disjoint sets:
  - **Few-shot set** (8 examples): Shown to judge once, never scored
  - **Calibration set** (40 examples): Judge scores repeatedly, gets feedback
  - **Golden set** (12 examples): Held-out validation, touched only once after calibration

Tier distribution per set:
- All sets are stratified across GOOD/OK/BAD tiers
- Dataset has 117 GOOD, 117 OK, 98 BAD examples for this style

### Judge (`judge.py`)

- Wraps Claude Sonnet 4 API calls
- Builds system prompt from rubric + few-shot examples
- Handles single grade calls: prompt in → {reasoning, grade} out
- Accumulates feedback context across calibration rounds
- Grades: GOOD / OK / BAD

### Harness (`harness.py`)

Main calibration script with:

**Calibration Loop:**
- Max 5 rounds
- Threshold: 90% exact match rate (default, configurable)
- **Reshuffles calibration set every round** (prevents positional memorization)
- Computes metrics: exact match rate, mean off-by, confusion matrix
- Builds plain-language feedback from confusion matrix
- Judge receives feedback for next round

**Golden Set Validation:**
- Run ONLY ONCE after calibration exits
- One-shot final check on held-out data
- Never seen during calibration

**Metrics Tracked:**
- Exact match rate
- Mean off-by (direction-agnostic)
- Cumulative off-by
- **Full confusion matrix** (preserves directional bias)

**Output:**
- Timestamped markdown report (`judge_report_YYYYMMDD_HHMMSS.md`)
- Timestamped JSON data (`judge_report_YYYYMMDD_HHMMSS.json`)

## Threshold Justification

**Default threshold: 0.9 (90% exact match rate)**

Rationale:
- With 3 tiers, random baseline is ~33%
- 90% is nearly 3× random, showing meaningful signal over chance
- Achievable given the balanced tier distribution (117/117/98)
- Represents strong agreement while allowing for edge cases

## Key Design Decisions

### 1. Calibration Set Reshuffling
The calibration set is **reshuffled every round** to prevent the judge from memorizing positional patterns rather than learning the grading standard.

### 2. Full Confusion Matrix
While we use a scalar "off-by" metric, we **track the full confusion matrix** to detect directional bias (e.g., systematically grading GOOD as OK vs. BAD as OK).

### 3. Golden Set Isolation
The golden set is **never touched until calibration exits** (whether by hitting threshold or max rounds). This prevents any form of overfitting to the validation set.

### 4. Timestamped Reports
Each run saves a **timestamped report** so you can compare performance across iterations when tweaking prompts or thresholds.

## Example Output

```
================================================================================
JUDGE CALIBRATION REPORT
================================================================================

Style graded: Chaos Cinema Critique  A Manifesto for the Absurd
Calibration set size: 40 | Golden set size: 12
Threshold: 75.0% exact match rate

Round-by-round performance:
  Round 1: exact_match=0.55, mean_off_by=0.60, cumulative_off_by=24
  Round 2: exact_match=0.68, mean_off_by=0.38, cumulative_off_by=15
  Round 3: exact_match=0.78, mean_off_by=0.25, cumulative_off_by=10  <- threshold met

Confusion matrix (final calibration round):
              pred:BAD  pred:OK  pred:GOOD
true:BAD         12        2         0
true:OK           1       15         2
true:GOOD         0        1        17

Golden set (held-out, one-shot):
  exact_match=0.75, mean_off_by=0.25

VERDICT:
  Judge improved +23% over 3 rounds (0.55 -> 0.78) on
  calibration set, and achieved 0.75 on unseen golden set.
  Judge is USABLE for downstream generator feedback.
```

## Scope

### What's Included
- LLM judge calibration and validation
- Prompt-based grading (no fine-tuning)
- Feedback loop with confusion matrix analysis
- Comprehensive metrics and reporting

### Explicitly Out of Scope
- Generator implementation (follow-on project)
- Fine-tuning the judge
- Automated reasoning validation (dataset doesn't support it)

## Files

- `data_prep.py` - Dataset loading and splitting
- `judge.py` - LLM judge wrapper
- `harness.py` - Main calibration harness
- `analyze_dataset.py` - Dataset exploration tool (helper)
- `judge_report_*.md` - Timestamped calibration reports
- `judge_report_*.json` - Detailed calibration data

## Running Custom Configurations

```python
# In harness.py or from command line:
from harness import main

# Custom threshold
main(threshold=0.80, max_rounds=5, seed=42)

# More rounds
main(threshold=0.75, max_rounds=10, seed=42)

# Different seed
main(threshold=0.75, max_rounds=5, seed=123)
```

## Notes

- The spec requires Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Dataset warnings about unauthenticated HF requests are harmless
- First run will download ~3MB dataset from HuggingFace
- Each calibration run makes ~40-200 API calls depending on rounds needed
