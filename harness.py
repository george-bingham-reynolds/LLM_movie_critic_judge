"""
harness.py - Main calibration harness for the LLM judge.

Runs the full calibration loop, computes metrics, provides feedback,
and validates on the golden set. Saves timestamped reports.

Cost-optimized version: score_round() now sends the whole round in a
SINGLE batched call via judge.grade_batch() instead of looping and
calling judge.grade() once per item. Combined with judge.py's prompt
caching on the static system prompt, this cuts both the number of API
calls and the cost of the repeated static content dramatically.
"""

import random
from collections import defaultdict
from datetime import datetime
import json

from data_prep import prepare_data, extract_rubric_name
from judge import Judge


def score_round(judge, calibration_set, round_num):
    """
    Score all examples in the calibration set (after shuffling), in ONE
    batched API call.

    Args:
        judge: Judge instance
        calibration_set: List of example dicts (must each have an 'id')
        round_num: Current round number

    Returns:
        List of prediction dicts with keys: id, true_tier, pred_tier, reasoning
    """
    # IMPORTANT: Reshuffle every round to avoid positional memorization
    shuffled = calibration_set.copy()
    random.shuffle(shuffled)

    print(f"\nRound {round_num}: Grading {len(shuffled)} examples in one batched call...")

    batch_results = judge.grade_batch(shuffled)
    results_by_id = {r['id']: r for r in batch_results}

    predictions = []
    for item in shuffled:
        item_id = str(item['id'])
        judge_output = results_by_id.get(item_id)

        if judge_output is None:
            # Shouldn't happen given judge.py's fallback, but guard anyway
            judge_output = {'reasoning': '[MISSING FROM BATCH RESPONSE]', 'grade': 'OK'}

        predictions.append({
            'id': item_id,
            'true_tier': item['tier'],
            'pred_tier': judge_output['grade'],
            'reasoning': judge_output['reasoning']
        })

    print(f"  Graded {len(predictions)}/{len(shuffled)} examples.")

    return predictions


def score_metrics(predictions):
    """
    Compute performance metrics from predictions.

    Args:
        predictions: List of dicts with true_tier and pred_tier

    Returns:
        dict with keys: cumulative_off_by, mean_off_by, confusion_matrix, exact_match_rate
    """
    tier_to_num = {"BAD": 0, "OK": 1, "GOOD": 2}

    off_by_total = 0
    confusion = defaultdict(int)
    exact_matches = 0

    for p in predictions:
        true_tier = p['true_tier']
        pred_tier = p['pred_tier']

        true_n = tier_to_num[true_tier]
        pred_n = tier_to_num[pred_tier]
        off_by_total += abs(true_n - pred_n)

        confusion[(true_tier, pred_tier)] += 1

        if true_tier == pred_tier:
            exact_matches += 1

    return {
        'cumulative_off_by': off_by_total,
        'mean_off_by': off_by_total / len(predictions),
        'confusion_matrix': dict(confusion),
        'exact_match_rate': exact_matches / len(predictions)
    }


def build_feedback_message(metrics):
    """
    Convert confusion matrix into plain-language feedback.

    Args:
        metrics: Dict with 'confusion_matrix' key

    Returns:
        Plain-language feedback string describing grading errors
    """
    confusion = metrics['confusion_matrix']
    feedback_parts = []

    errors = []
    for true_tier in ['GOOD', 'OK', 'BAD']:
        for pred_tier in ['GOOD', 'OK', 'BAD']:
            if true_tier != pred_tier:
                count = confusion.get((true_tier, pred_tier), 0)
                if count > 0:
                    errors.append((count, true_tier, pred_tier))

    errors.sort(reverse=True)

    if not errors:
        return "Your grading was perfect this round! All examples were graded correctly."

    feedback_parts.append("Grading errors from this round:")
    feedback_parts.append("")

    for count, true_tier, pred_tier in errors:
        if count == 1:
            feedback_parts.append(f"  • You graded {count} {true_tier} item as {pred_tier}")
        else:
            feedback_parts.append(f"  • You graded {count} {true_tier} items as {pred_tier}")

    feedback_parts.append("")

    # if any(true == 'GOOD' and pred in ['OK', 'BAD'] for _, true, pred in errors):
    #     feedback_parts.append("You seem to be grading GOOD examples too harshly. Look for strong adherence to the rubric's style markers.")

    # if any(true == 'BAD' and pred in ['OK', 'GOOD'] for _, true, pred in errors):
    #     feedback_parts.append("You seem to be grading BAD examples too leniently. These should clearly fail to match the rubric.")

    # if any(true == 'OK' for _, true, pred in errors):
    #     feedback_parts.append("Remember: OK means partial match — some stylistic elements present but not fully realized.")

    return "\n".join(feedback_parts)


def print_confusion_matrix(confusion):
    """Pretty-print the confusion matrix."""
    tiers = ['BAD', 'OK', 'GOOD']

    print(f"              {'pred:BAD':>10}  {'pred:OK':>10}  {'pred:GOOD':>10}")

    for true_tier in tiers:
        row_str = f"true:{true_tier:<5}"
        for pred_tier in tiers:
            count = confusion.get((true_tier, pred_tier), 0)
            row_str += f"  {count:>10}"
        print(row_str)


def run_calibration(judge, calibration_set, max_rounds=5, threshold=0.9):
    """
    Run the calibration loop with feedback.

    Args:
        judge: Judge instance
        calibration_set: List of calibration examples
        max_rounds: Maximum number of calibration rounds
        threshold: Exact match rate threshold to exit calibration

    Returns:
        List of round history dicts
    """
    print(f"\n{'='*80}")
    print("CALIBRATION LOOP")
    print(f"{'='*80}")
    print(f"Max rounds: {max_rounds}")
    print(f"Threshold for success: {threshold:.1%} exact match rate")
    print(f"Calibration set size: {len(calibration_set)}")
    print(f"(Each round = 1 batched API call, not {len(calibration_set)} individual calls)")

    round_history = []

    for round_num in range(1, max_rounds + 1):
        predictions = score_round(judge, calibration_set, round_num)
        metrics = score_metrics(predictions)

        round_history.append({
            'round': round_num,
            'predictions': predictions,
            **metrics
        })

        print(f"\nRound {round_num} results:")
        print(f"  Exact match rate: {metrics['exact_match_rate']:.2%}")
        print(f"  Mean off-by: {metrics['mean_off_by']:.2f}")
        print(f"  Cumulative off-by: {metrics['cumulative_off_by']}")
        print(f"\n  Confusion matrix:")
        print_confusion_matrix(metrics['confusion_matrix'])

        if metrics['exact_match_rate'] >= threshold:
            print(f"\n✓ Threshold of {threshold:.1%} met! Calibration complete.")
            break

        if round_num < max_rounds:
            feedback = build_feedback_message(metrics)
            print(f"\n  Feedback for next round:")
            for line in feedback.split('\n'):
                print(f"    {line}")
            judge.receive_feedback(feedback)
        else:
            print(f"\n✗ Max rounds reached without meeting threshold.")

    return round_history


def run_golden_check(judge, golden_set):
    """
    Run the final validation on the held-out golden set.

    Args:
        judge: Calibrated judge instance
        golden_set: List of golden examples

    Returns:
        dict with predictions and metrics
    """
    print(f"\n{'='*80}")
    print("GOLDEN SET VALIDATION (one-shot, held-out)")
    print(f"{'='*80}")

    predictions = score_round(judge, golden_set, "golden")
    metrics = score_metrics(predictions)

    print(f"\nGolden set results:")
    print(f"  Exact match rate: {metrics['exact_match_rate']:.2%}")
    print(f"  Mean off-by: {metrics['mean_off_by']:.2f}")
    print(f"\n  Confusion matrix:")
    print_confusion_matrix(metrics['confusion_matrix'])

    return {
        'predictions': predictions,
        **metrics
    }


def generate_report(style_name, calibration_history, golden_results,
                   calibration_size, golden_size, threshold):
    """
    Generate the final report text.

    Returns:
        String containing the formatted report
    """
    lines = []
    lines.append("="*80)
    lines.append("JUDGE CALIBRATION REPORT")
    lines.append("="*80)
    lines.append("")
    lines.append(f"Style graded: {style_name}")
    lines.append(f"Calibration set size: {calibration_size} | Golden set size: {golden_size}")
    lines.append(f"Threshold: {threshold:.1%} exact match rate")
    lines.append("")
    lines.append("Round-by-round performance:")

    for round_data in calibration_history:
        r = round_data['round']
        em = round_data['exact_match_rate']
        moff = round_data['mean_off_by']
        coff = round_data['cumulative_off_by']

        marker = " <- threshold met" if em >= threshold else ""
        lines.append(f"  Round {r}: exact_match={em:.2f}, mean_off_by={moff:.2f}, cumulative_off_by={coff}{marker}")

    final_round = calibration_history[-1]
    lines.append("")
    lines.append("Confusion matrix (final calibration round):")

    confusion = final_round['confusion_matrix']
    lines.append(f"              {'pred:BAD':>10}  {'pred:OK':>10}  {'pred:GOOD':>10}")
    for true_tier in ['BAD', 'OK', 'GOOD']:
        row_str = f"true:{true_tier:<5}"
        for pred_tier in ['BAD', 'OK', 'GOOD']:
            count = confusion.get((true_tier, pred_tier), 0)
            row_str += f"  {count:>10}"
        lines.append(row_str)

    lines.append("")
    lines.append("Golden set (held-out, one-shot):")
    lines.append(f"  exact_match={golden_results['exact_match_rate']:.2f}, mean_off_by={golden_results['mean_off_by']:.2f}")

    lines.append("")
    lines.append("VERDICT:")

    first_round = calibration_history[0]
    final_calib = calibration_history[-1]

    improvement = final_calib['exact_match_rate'] - first_round['exact_match_rate']
    num_rounds = len(calibration_history)

    lines.append(f"  Judge improved {improvement:+.0%} over {num_rounds} rounds "
                f"({first_round['exact_match_rate']:.2f} -> {final_calib['exact_match_rate']:.2f}) on")
    lines.append(f"  calibration set, and achieved {golden_results['exact_match_rate']:.2f} on unseen golden set.")

    if final_calib['exact_match_rate'] >= threshold and golden_results['exact_match_rate'] >= threshold * 0.9:
        lines.append(f"  Judge is USABLE for downstream generator feedback.")
    elif golden_results['exact_match_rate'] < final_calib['exact_match_rate'] - 0.1:
        lines.append(f"  NOTE: Golden performance degraded vs calibration — possible overfitting to feedback.")
    else:
        lines.append(f"  Judge needs further calibration or threshold adjustment.")

    lines.append("")
    lines.append("="*80)

    return "\n".join(lines)


def save_report(report_text, calibration_history, golden_results):
    """Save report to timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_filename = f"judge_report_{timestamp}.md"
    with open(md_filename, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {md_filename}")

    # Convert confusion matrix tuple keys to strings for JSON serialization
    def convert_confusion_matrix(data):
        """Recursively convert tuple keys in confusion matrices to strings."""
        if isinstance(data, dict):
            new_dict = {}
            for key, value in data.items():
                if key == 'confusion_matrix' and isinstance(value, dict):
                    # Convert tuple keys to "true->pred" format
                    new_dict[key] = {f"{k[0]}->{k[1]}": v for k, v in value.items()}
                else:
                    new_dict[key] = convert_confusion_matrix(value)
            return new_dict
        elif isinstance(data, list):
            return [convert_confusion_matrix(item) for item in data]
        else:
            return data
    
    json_filename = f"judge_report_{timestamp}.json"
    json_data = convert_confusion_matrix({
        'timestamp': timestamp,
        'calibration_history': calibration_history,
        'golden_results': golden_results
    })
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Detailed data saved to: {json_filename}")


def main(threshold=0.9, max_rounds=5, seed=42):
    """
    Main harness entry point.

    Args:
        threshold: Exact match rate threshold for calibration exit
        max_rounds: Maximum calibration rounds
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("="*80)
    print("LLM JUDGE CALIBRATION HARNESS")
    print("="*80)

    print("\nLoading and preparing data...")
    few_shot_set, calibration_set, golden_set, rubric_text = prepare_data(seed=seed)
    style_name = extract_rubric_name(rubric_text)

    print(f"\nInitializing judge for style: {style_name}")
    judge = Judge(
        rubric_text=rubric_text,
        style_name=style_name,
        few_shot_examples=few_shot_set
    )

    calibration_history = run_calibration(
        judge=judge,
        calibration_set=calibration_set,
        max_rounds=max_rounds,
        threshold=threshold
    )

    golden_results = run_golden_check(judge, golden_set)

    print(f"\n{'='*80}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*80}")

    report_text = generate_report(
        style_name=style_name,
        calibration_history=calibration_history,
        golden_results=golden_results,
        calibration_size=len(calibration_set),
        golden_size=len(golden_set),
        threshold=threshold
    )

    print("\n" + report_text)

    save_report(report_text, calibration_history, golden_results)

    print("\n" + "="*80)
    print("CALIBRATION COMPLETE!")
    print("="*80)


# ============================================================================
# GENERATOR PHASE (Phase 2)
# ============================================================================

from generator import Generator
from generator_prompts import prepare_generator_data, print_data_assignment_table


def save_generations_to_file(generations, round_num):
    """
    Save generated reviews to human-readable files for manual spot-checking.
    
    Args:
        generations: List of dicts with keys: prompt_id, prompt, generated_review
        round_num: Round number (or "held_out")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as markdown for easy reading
    md_filename = f"generations_round_{round_num}_{timestamp}.md"
    with open(md_filename, 'w') as f:
        f.write(f"# Generated Reviews - Round {round_num}\n")
        f.write(f"Generated at: {timestamp}\n\n")
        f.write("="*80 + "\n\n")
        
        for gen in generations:
            f.write(f"## Prompt ID: {gen['prompt_id']}\n\n")
            f.write(f"**Topic:** {gen['prompt']}\n\n")
            f.write(f"**Generated Review:**\n\n")
            f.write(gen['generated_review'])
            f.write("\n\n" + "-"*80 + "\n\n")
    
    print(f"  Saved generated reviews to: {md_filename}")
    
    # Also save as JSON for programmatic access
    json_filename = f"generations_round_{round_num}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(generations, f, indent=2)


def compute_average_grade(judge_results):
    """
    Compute average numeric score from judge grades.
    
    Maps: GOOD=2, OK=1, BAD=0, then averages.
    
    Args:
        judge_results: List of dicts with 'grade' key
        
    Returns:
        float: Average score (0.0 to 2.0)
    """
    tier_to_num = {"BAD": 0, "OK": 1, "GOOD": 2}
    scores = [tier_to_num[r['grade']] for r in judge_results]
    return sum(scores) / len(scores) if scores else 0.0


def build_generator_feedback(generations, judge_results):
    """
    Build per-item feedback for generator from judge's grades and reasoning.
    
    Spec: Per-item detail (not aggregate) since aggregate feedback showed flat
    improvement in judge's own calibration.
    
    Args:
        generations: List of dicts with prompt_id, prompt, generated_review
        judge_results: List of dicts with id, grade, reasoning
        
    Returns:
        String containing per-item feedback
    """
    judge_by_id = {r['id']: r for r in judge_results}
    
    feedback_parts = []
    
    for gen in generations:
        judge_result = judge_by_id.get(gen['prompt_id'])
        if not judge_result:
            continue
        
        grade = judge_result['grade']
        reasoning = judge_result['reasoning']
        
        # Format as concrete, actionable feedback
        if grade == 'BAD':
            feedback_parts.append(
                f"Prompt {gen['prompt_id']} ('{gen['prompt'][:40]}...'): "
                f"NEEDS MAJOR IMPROVEMENT (graded BAD). Judge's critique: {reasoning}"
            )
        elif grade == 'OK':
            feedback_parts.append(
                f"Prompt {gen['prompt_id']} ('{gen['prompt'][:40]}...'): "
                f"PARTIAL MATCH (graded OK). Judge's notes: {reasoning}"
            )
        else:  # GOOD
            feedback_parts.append(
                f"Prompt {gen['prompt_id']} ('{gen['prompt'][:40]}...'): "
                f"STRONG MATCH (graded GOOD). Keep this quality."
            )
    
    return "\n\n".join(feedback_parts)


def run_generator_phase(judge, judge_few_shot, judge_calibration, judge_golden, 
                       rubric_text, style_name, seed=42, max_rounds=3):
    """
    Run the generator feedback loop (Phase 2).
    
    Args:
        judge: Calibrated judge instance
        judge_few_shot: Judge's few-shot set (off-limits for generator)
        judge_calibration: Judge's calibration set (available pool)
        judge_golden: Judge's golden set (available pool)
        rubric_text: Style rubric
        style_name: Style name
        seed: Random seed
        max_rounds: Max generator rounds (default 3)
        
    Returns:
        dict with round_history and held_out results
    """
    print(f"\n{'='*80}")
    print("GENERATOR PHASE - ITERATIVE STYLE LEARNING")
    print(f"{'='*80}\n")
    
    # Prepare generator data
    gen_few_shot, gen_tracking, gen_held_out = prepare_generator_data(
        judge_few_shot, judge_calibration, judge_golden, seed=seed
    )
    
    # Print assignment table (required by spec)
    print_data_assignment_table(
        judge_few_shot, judge_calibration, judge_golden,
        gen_few_shot, gen_tracking, gen_held_out
    )
    
    # Initialize generator
    print(f"Initializing generator for style: {style_name}\n")
    generator = Generator(
        rubric_text=rubric_text,
        style_name=style_name,
        few_shot_examples=gen_few_shot
    )
    
    # Generator feedback loop
    print(f"{'='*80}")
    print(f"GENERATOR FEEDBACK LOOP (max {max_rounds} rounds)")
    print(f"{'='*80}")
    print(f"Round-tracking prompts: {len(gen_tracking)} (fixed, same every round)")
    print(f"Held-out prompts: {len(gen_held_out)} (used once at end)\n")
    
    round_history = []
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'-'*80}")
        print(f"Generator Round {round_num}")
        print(f"{'-'*80}")
        
        # 1. Generator writes reviews
        print(f"\nStep 1: Generating reviews for {len(gen_tracking)} prompts...")
        generations = generator.generate_batch(gen_tracking)
        
        # 2. Save generated text
        print(f"Step 2: Saving generated reviews...")
        save_generations_to_file(generations, round_num)
        
        # 3. Judge scores generated reviews
        print(f"Step 3: Judge scoring generated reviews...")
        judge_input = [
            {"id": g["prompt_id"], "prompt": g["prompt"], "response_text": g["generated_review"]}
            for g in generations
        ]
        judge_results = judge.grade_batch(judge_input)
        
        # 4. Compute average score
        round_avg_score = compute_average_grade(judge_results)
        
        # Count grades
        grade_counts = defaultdict(int)
        for r in judge_results:
            grade_counts[r['grade']] += 1
        
        round_history.append({
            'round': round_num,
            'avg_score': round_avg_score,
            'grade_counts': dict(grade_counts),
            'individual_grades': judge_results,
            'generations': generations
        })
        
        print(f"\nRound {round_num} results:")
        print(f"  Average judge score: {round_avg_score:.2f} / 2.0 ({round_avg_score/2.0:.1%})")
        print(f"  Grade distribution: GOOD={grade_counts['GOOD']}, OK={grade_counts['OK']}, BAD={grade_counts['BAD']}")
        
        # 5. Build and deliver per-item feedback
        if round_num < max_rounds:
            print(f"\nStep 4: Building per-item feedback for next round...")
            feedback = build_generator_feedback(generations, judge_results)
            print(f"\n  Feedback for next round:")
            for line in feedback.split('\n'):
                print(f"    {line}")
            generator.receive_feedback(feedback)
    
    # Final round: held-out prompts (one-shot, no feedback after)
    print(f"\n{'='*80}")
    print(f"GENERATOR HELD-OUT CHECK (generalization test)")
    print(f"{'='*80}\n")
    
    print(f"Generating reviews for {len(gen_held_out)} held-out prompts...")
    held_out_generations = generator.generate_batch(gen_held_out)
    
    print(f"Saving held-out generations...")
    save_generations_to_file(held_out_generations, "held_out")
    
    print(f"Judge scoring held-out generations...")
    held_out_judge_input = [
        {"id": g["prompt_id"], "prompt": g["prompt"], "response_text": g["generated_review"]}
        for g in held_out_generations
    ]
    held_out_judge_results = judge.grade_batch(held_out_judge_input)
    
    held_out_avg_score = compute_average_grade(held_out_judge_results)
    held_out_grade_counts = defaultdict(int)
    for r in held_out_judge_results:
        held_out_grade_counts[r['grade']] += 1
    
    print(f"\nHeld-out results:")
    print(f"  Average judge score: {held_out_avg_score:.2f} / 2.0 ({held_out_avg_score/2.0:.1%})")
    print(f"  Grade distribution: GOOD={held_out_grade_counts['GOOD']}, OK={held_out_grade_counts['OK']}, BAD={held_out_grade_counts['BAD']}")
    
    final_round_score = round_history[-1]['avg_score']
    gap = final_round_score - held_out_avg_score
    
    print(f"\nGeneralization check:")
    print(f"  Final round-tracking score: {final_round_score:.2f}")
    print(f"  Held-out score: {held_out_avg_score:.2f}")
    print(f"  Gap: {gap:+.2f} {'(good generalization)' if abs(gap) < 0.3 else '(possible prompt-specific patching)'}")
    
    return {
        'round_history': round_history,
        'held_out': {
            'avg_score': held_out_avg_score,
            'grade_counts': dict(held_out_grade_counts),
            'individual_grades': held_out_judge_results,
            'generations': held_out_generations
        }
    }


def main(threshold=0.9, max_rounds=5, seed=42, generator_trigger_threshold=0.80):
    """
    Main harness entry point - runs judge calibration, then generator phase if triggered.

    Args:
        threshold: Judge exact match rate threshold for calibration exit
        max_rounds: Maximum judge calibration rounds
        seed: Random seed for reproducibility
        generator_trigger_threshold: Golden-set threshold to trigger generator phase
    """
    random.seed(seed)

    print("="*80)
    print("LLM JUDGE CALIBRATION HARNESS - PHASE 1 & 2")
    print("="*80)

    # ===== PHASE 1: JUDGE CALIBRATION =====
    print("\n" + "="*80)
    print("PHASE 1: JUDGE CALIBRATION")
    print("="*80)
    
    print("\nLoading and preparing data...")
    few_shot_set, calibration_set, golden_set, rubric_text = prepare_data(seed=seed)
    style_name = extract_rubric_name(rubric_text)

    print(f"\nInitializing judge for style: {style_name}")
    judge = Judge(
        rubric_text=rubric_text,
        style_name=style_name,
        few_shot_examples=few_shot_set
    )

    calibration_history = run_calibration(
        judge=judge,
        calibration_set=calibration_set,
        max_rounds=max_rounds,
        threshold=threshold
    )

    golden_results = run_golden_check(judge, golden_set)

    print(f"\n{'='*80}")
    print("GENERATING PHASE 1 REPORT")
    print(f"{'='*80}")

    report_text = generate_report(
        style_name=style_name,
        calibration_history=calibration_history,
        golden_results=golden_results,
        calibration_size=len(calibration_set),
        golden_size=len(golden_set),
        threshold=threshold
    )

    print("\n" + report_text)

    save_report(report_text, calibration_history, golden_results)

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE!")
    print("="*80)
    
    # ===== PHASE 2: GENERATOR (golden-gated trigger) =====
    print(f"\n{'='*80}")
    print(f"PHASE 2: GENERATOR TRIGGER CHECK")
    print(f"{'='*80}")
    
    golden_performance = golden_results['exact_match_rate']
    
    if golden_performance >= generator_trigger_threshold:
        print(f"\n✓ Golden set performance ({golden_performance:.1%}) cleared trigger threshold ({generator_trigger_threshold:.0%}).")
        print(f"  Proceeding to generator phase...\n")
        
        generator_results = run_generator_phase(
            judge=judge,
            judge_few_shot=few_shot_set,
            judge_calibration=calibration_set,
            judge_golden=golden_set,
            rubric_text=rubric_text,
            style_name=style_name,
            seed=seed,
            max_rounds=3
        )
        
        print(f"\n{'='*80}")
        print("PHASE 2 COMPLETE!")
        print("="*80)
        
        # Print summary
        print(f"\nGenerator improvement summary:")
        for round_data in generator_results['round_history']:
            print(f"  Round {round_data['round']}: avg score = {round_data['avg_score']:.2f}")
        print(f"  Held-out: avg score = {generator_results['held_out']['avg_score']:.2f}")
        
    else:
        print(f"\n✗ Golden set performance ({golden_performance:.1%}) below trigger threshold ({generator_trigger_threshold:.0%}).")
        print(f"  Generator phase skipped. Judge needs further calibration first.")
    
    print(f"\n{'='*80}")
    print("ALL PHASES COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    # Default threshold: 0.9 for judge calibration
    # Generator trigger: 0.80 (lower than judge target, but shows judge is working)
    # Justification: With 3 tiers, random baseline is ~33%.
    # 90% exact match is nearly 3x random, showing meaningful signal.
    # 80% trigger ensures judge is validated before generator runs.

    main(threshold=0.9, max_rounds=5, seed=42, generator_trigger_threshold=0.80)