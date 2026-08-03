"""
data_prep.py - Load and prepare the creative-rubrics dataset for judge calibration.

Loads vicgalle/creative-rubrics, filters to "Chaos Cinema Critique" style,
and splits into three disjoint sets: few_shot, calibration, and golden.
"""

from datasets import load_dataset
from collections import defaultdict
import random


def map_qualifier_to_tier(qualifier):
    """Map dataset qualifier to GOOD/OK/BAD tier."""
    qualifier_lower = qualifier.lower()
    if 'high' in qualifier_lower:
        return 'GOOD'
    elif 'low' in qualifier_lower:
        return 'BAD'
    else:  # '50/100 score'
        return 'OK'


def extract_rubric_name(rubric_text):
    """Extract clean rubric name from rubric text."""
    first_line = rubric_text.split('\n')[0].strip()
    return first_line.replace('The "', '').replace('"', '').replace('Anti-Rubric:', '').strip()


def load_and_filter_dataset(target_style="Chaos Cinema Critique", seed=42):
    """
    Load the dataset and filter to a single rubric style.
    
    Args:
        target_style: Name (or substring) of the rubric style to filter for
        seed: Random seed for reproducibility
        
    Returns:
        List of dicts with keys: id, prompt, response_text, rubric, tier
    """
    print(f"Loading dataset from HuggingFace...")
    dataset = load_dataset("vicgalle/creative-rubrics")
    
    # Filter to target style
    filtered_examples = []
    for idx, row in enumerate(dataset['train']):
        rubric_name = extract_rubric_name(row['rubric'])
        if target_style.lower() in rubric_name.lower():
            filtered_examples.append({
                'id': f"ex_{idx}",
                'prompt': row['prompt'],
                'response_text': row['response'],
                'rubric': row['rubric'],
                'tier': map_qualifier_to_tier(row['qualifier'])
            })
    
    print(f"Filtered to style '{target_style}': {len(filtered_examples)} examples")
    
    # Show tier distribution
    tier_counts = defaultdict(int)
    for ex in filtered_examples:
        tier_counts[ex['tier']] += 1
    print(f"Tier distribution: GOOD={tier_counts['GOOD']}, OK={tier_counts['OK']}, BAD={tier_counts['BAD']}")
    
    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(filtered_examples)
    
    return filtered_examples


def stratified_split(examples, few_shot_size=8, calibration_size=40, golden_size=12):
    """
    Split examples into three disjoint, stratified sets.
    
    Args:
        examples: List of example dicts with 'tier' key
        few_shot_size: Number of examples for few-shot set
        calibration_size: Number of examples for calibration set
        golden_size: Number of examples for golden set
        
    Returns:
        tuple: (few_shot_set, calibration_set, golden_set)
    """
    # Group by tier
    by_tier = defaultdict(list)
    for ex in examples:
        by_tier[ex['tier']].append(ex)
    
    # Calculate proportional splits for each tier
    total_needed = few_shot_size + calibration_size + golden_size
    
    few_shot_set = []
    calibration_set = []
    golden_set = []
    
    for tier in ['GOOD', 'OK', 'BAD']:
        tier_examples = by_tier[tier]
        tier_total = len(tier_examples)
        
        # Calculate proportional sizes (rounded)
        tier_few_shot = max(1, round(few_shot_size * tier_total / sum(len(by_tier[t]) for t in ['GOOD', 'OK', 'BAD'])))
        tier_calibration = max(1, round(calibration_size * tier_total / sum(len(by_tier[t]) for t in ['GOOD', 'OK', 'BAD'])))
        tier_golden = max(1, round(golden_size * tier_total / sum(len(by_tier[t]) for t in ['GOOD', 'OK', 'BAD'])))
        
        # Ensure we don't exceed available examples
        tier_few_shot = min(tier_few_shot, tier_total)
        tier_calibration = min(tier_calibration, tier_total - tier_few_shot)
        tier_golden = min(tier_golden, tier_total - tier_few_shot - tier_calibration)
        
        # Split
        few_shot_set.extend(tier_examples[:tier_few_shot])
        calibration_set.extend(tier_examples[tier_few_shot:tier_few_shot + tier_calibration])
        golden_set.extend(tier_examples[tier_few_shot + tier_calibration:tier_few_shot + tier_calibration + tier_golden])
    
    print(f"\nSplit results:")
    print(f"  Few-shot set: {len(few_shot_set)} examples")
    print(f"  Calibration set: {len(calibration_set)} examples")
    print(f"  Golden set: {len(golden_set)} examples")
    
    # Verify stratification
    for set_name, dataset in [("Few-shot", few_shot_set), ("Calibration", calibration_set), ("Golden", golden_set)]:
        tier_counts = defaultdict(int)
        for ex in dataset:
            tier_counts[ex['tier']] += 1
        print(f"  {set_name} tiers: GOOD={tier_counts['GOOD']}, OK={tier_counts['OK']}, BAD={tier_counts['BAD']}")
    
    return few_shot_set, calibration_set, golden_set


def prepare_data(target_style="Chaos Cinema Critique", seed=42):
    """
    Main function to load and prepare all data splits.
    
    Returns:
        tuple: (few_shot_set, calibration_set, golden_set, rubric_text)
    """
    # Load and filter
    examples = load_and_filter_dataset(target_style=target_style, seed=seed)
    
    # Split
    few_shot_set, calibration_set, golden_set = stratified_split(
        examples,
        few_shot_size=8,
        calibration_size=40,
        golden_size=12
    )
    
    # Extract rubric text (same for all examples in this style)
    rubric_text = examples[0]['rubric']
    
    return few_shot_set, calibration_set, golden_set, rubric_text


if __name__ == "__main__":
    # Test the data preparation
    few_shot, calibration, golden, rubric = prepare_data()
    
    print(f"\n{'='*80}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Rubric: {extract_rubric_name(rubric)}")
    print(f"Ready for judge calibration!")
