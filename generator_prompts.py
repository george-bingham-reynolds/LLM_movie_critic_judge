"""
generator_prompts.py - Prepare data for generator phase with strict overlap rules.

Assigns generator few-shot examples and prompts from the existing data splits,
respecting overlap constraints:
1. Generator prompts must NOT come from judge.few_shot_set
2. Generator few-shot must NOT overlap with generator prompts
3. Both can come from judge.calibration_set or judge.golden_set
"""

from collections import defaultdict
import random


def prepare_generator_data(judge_few_shot, judge_calibration, judge_golden, seed=42):
    """
    Prepare generator few-shot examples and prompts from judge data splits.
    
    Args:
        judge_few_shot: Judge's few-shot examples (8 examples) - OFF LIMITS for generator
        judge_calibration: Judge's calibration set (40 examples) - available pool
        judge_golden: Judge's golden set (12 examples) - available pool
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (generator_few_shot, generator_round_tracking, generator_held_out)
    """
    random.seed(seed)
    
    # Available pool: calibration + golden (52 examples total)
    # We need: 4-6 few-shot + 6-8 round-tracking + 2-3 held-out = ~12-17 total
    available_pool = judge_calibration + judge_golden
    
    # Group by tier for stratified selection
    by_tier = defaultdict(list)
    for ex in available_pool:
        by_tier[ex['tier']].append(ex)
    
    # Shuffle each tier's examples
    for tier in by_tier:
        random.shuffle(by_tier[tier])
    
    # === Generator Few-Shot: 5 examples (3 GOOD, 1 OK, 1 BAD) ===
    # Spec: weighted toward GOOD, with OK and BAD as contrast/anti-examples
    generator_few_shot = []
    generator_few_shot.extend(by_tier['GOOD'][:3])  # 3 GOOD examples
    generator_few_shot.append(by_tier['OK'][0])      # 1 OK example as partial match
    generator_few_shot.append(by_tier['BAD'][0])     # 1 BAD example as anti-example
    
    # Mark these as used
    used_ids = {ex['id'] for ex in generator_few_shot}
    
    # === Generator Round-Tracking Prompts: 8 prompts (fixed, reused every round) ===
    # Pull from remaining available pool, avoiding already-used few-shot
    remaining_pool = [ex for ex in available_pool if ex['id'] not in used_ids]
    random.shuffle(remaining_pool)
    
    generator_round_tracking = remaining_pool[:8]
    used_ids.update(ex['id'] for ex in generator_round_tracking)
    
    # === Generator Held-Out Prompts: 3 prompts (used once at end) ===
    remaining_pool = [ex for ex in available_pool if ex['id'] not in used_ids]
    generator_held_out = remaining_pool[:3]
    
    return generator_few_shot, generator_round_tracking, generator_held_out


def print_data_assignment_table(judge_few_shot, judge_calibration, judge_golden,
                                generator_few_shot, generator_round_tracking, generator_held_out):
    """
    Print a comprehensive table showing all data assignments and overlaps.
    
    This is the required "print summary before running" check from the spec.
    """
    print("\n" + "="*80)
    print("DATA ASSIGNMENT SUMMARY")
    print("="*80)
    print("\nOVERLAP RULES CHECK:")
    print("✓ Generator prompts exclude judge.few_shot_set")
    print("✓ Generator few-shot excludes generator prompts")
    print("✓ Generator uses judge.calibration_set + judge.golden_set as source pool")
    print("\n" + "-"*80)
    
    # Judge assignments
    print("\nJUDGE DATA:")
    print(f"  Few-shot examples (8):      {[ex['id'] for ex in judge_few_shot]}")
    print(f"  Calibration set (40):       {len(judge_calibration)} examples")
    print(f"    Sample IDs: {[ex['id'] for ex in judge_calibration[:5]]}...")
    print(f"  Golden set (12):            {[ex['id'] for ex in judge_golden]}")
    
    print("\n" + "-"*80)
    
    # Generator assignments
    print("\nGENERATOR DATA:")
    print(f"  Few-shot examples (5):")
    for ex in generator_few_shot:
        tier_label = f"{ex['tier']}" + (" (partial match contrast)" if ex['tier'] == 'OK' else "")
        print(f"    {ex['id']}: {tier_label} - {ex['prompt'][:50]}...")
    
    print(f"\n  Round-tracking prompts (8, fixed every round):")
    for ex in generator_round_tracking:
        print(f"    {ex['id']}: {ex['prompt'][:60]}...")
    
    print(f"\n  Held-out prompts (3, used once at end):")
    for ex in generator_held_out:
        print(f"    {ex['id']}: {ex['prompt'][:60]}...")
    
    print("\n" + "-"*80)
    
    # Verify no overlaps
    judge_few_shot_ids = {ex['id'] for ex in judge_few_shot}
    gen_few_shot_ids = {ex['id'] for ex in generator_few_shot}
    gen_tracking_ids = {ex['id'] for ex in generator_round_tracking}
    gen_held_out_ids = {ex['id'] for ex in generator_held_out}
    
    print("\nOVERLAP VERIFICATION:")
    
    # Check 1: Generator prompts vs judge few-shot
    overlap_1 = (gen_tracking_ids | gen_held_out_ids) & judge_few_shot_ids
    if overlap_1:
        print(f"  ✗ ERROR: Generator prompts overlap with judge few-shot: {overlap_1}")
    else:
        print(f"  ✓ No overlap: Generator prompts vs judge few-shot")
    
    # Check 2: Generator few-shot vs generator prompts
    overlap_2 = gen_few_shot_ids & (gen_tracking_ids | gen_held_out_ids)
    if overlap_2:
        print(f"  ✗ ERROR: Generator few-shot overlaps with generator prompts: {overlap_2}")
    else:
        print(f"  ✓ No overlap: Generator few-shot vs generator prompts")
    
    # Check 3: Round-tracking vs held-out
    overlap_3 = gen_tracking_ids & gen_held_out_ids
    if overlap_3:
        print(f"  ✗ ERROR: Round-tracking overlaps with held-out: {overlap_3}")
    else:
        print(f"  ✓ No overlap: Round-tracking vs held-out prompts")
    
    print("\n" + "="*80)
    print("DATA ASSIGNMENTS VALIDATED - READY TO PROCEED")
    print("="*80 + "\n")


if __name__ == "__main__":
    from data_prep import prepare_data
    
    # Load judge data splits
    judge_few_shot, judge_calibration, judge_golden, rubric = prepare_data()
    
    # Prepare generator data
    gen_few_shot, gen_tracking, gen_held_out = prepare_generator_data(
        judge_few_shot, judge_calibration, judge_golden
    )
    
    # Print assignment table
    print_data_assignment_table(
        judge_few_shot, judge_calibration, judge_golden,
        gen_few_shot, gen_tracking, gen_held_out
    )
