"""
Quick script to analyze vicgalle/creative-rubrics dataset
and determine which style has best tier coverage.
"""

from datasets import load_dataset
from collections import Counter, defaultdict

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("vicgalle/creative-rubrics")

# Inspect the structure
print(f"\nDataset structure: {dataset}")
print(f"\nSample row keys: {list(dataset['train'][0].keys())}")
print(f"\nFirst row example:")
for key, value in dataset['train'][0].items():
    print(f"  {key}: {str(value)[:200]}")

# Analyze qualifiers (tiers)
print("\n" + "="*80)
print("UNIQUE QUALIFIERS (TIERS)")
print("="*80)
qualifiers = set(row['qualifier'] for row in dataset['train'])
for q in sorted(qualifiers):
    print(f"  {q}")

# Map qualifiers to tier labels
# Based on the spec: "highest / 50-100 / lowest" or similar
def map_qualifier_to_tier(qualifier):
    qualifier_lower = qualifier.lower()
    if 'high' in qualifier_lower or 'good' in qualifier_lower or 'excellent' in qualifier_lower:
        return 'GOOD'
    elif 'low' in qualifier_lower or 'poor' in qualifier_lower or 'bad' in qualifier_lower:
        return 'BAD'
    else:
        return 'OK'

# Extract unique rubric names (first line usually contains the name)
def extract_rubric_name(rubric_text):
    # Take the first line as the rubric name
    first_line = rubric_text.split('\n')[0].strip()
    # Remove common prefixes
    first_line = first_line.replace('The "', '').replace('"', '').replace('Anti-Rubric:', '').replace('Rubric:', '').strip()
    # Take first 50 chars max for display
    return first_line[:60]

# Analyze styles and tiers
print("\n" + "="*80)
print("TIER DISTRIBUTION BY RUBRIC STYLE")
print("="*80)

style_tier_counts = defaultdict(lambda: Counter())

for row in dataset['train']:
    rubric_name = extract_rubric_name(row['rubric'])
    tier = map_qualifier_to_tier(row['qualifier'])
    style_tier_counts[rubric_name][tier] += 1

# Sort by total count
sorted_styles = sorted(style_tier_counts.items(), 
                      key=lambda x: sum(x[1].values()), 
                      reverse=True)

print(f"\nFound {len(sorted_styles)} unique rubric styles:\n")
for rubric_name, tier_counts in sorted_styles:
    total = sum(tier_counts.values())
    good = tier_counts['GOOD']
    ok = tier_counts['OK']
    bad = tier_counts['BAD']
    
    # Check if style has coverage in all three tiers
    has_full_coverage = good > 0 and ok > 0 and bad > 0
    coverage_mark = "✓" if has_full_coverage else "✗"
    
    print(f"{coverage_mark} {rubric_name}")
    print(f"   Total: {total:3d} | GOOD: {good:3d} | OK: {ok:3d} | BAD: {bad:3d}")
    print()

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nStyles with full tier coverage (all three tiers present):")
for rubric_name, tier_counts in sorted_styles:
    good = tier_counts['GOOD']
    ok = tier_counts['OK']
    bad = tier_counts['BAD']
    if good > 0 and ok > 0 and bad > 0:
        total = sum(tier_counts.values())
        print(f"  • {rubric_name}: {total} examples ({good} GOOD, {ok} OK, {bad} BAD)")
