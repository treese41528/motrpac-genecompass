#!/usr/bin/env python3
"""
Analyze LLM study analysis output to assess value.
"""

import json
import sys
from collections import Counter

def analyze_output(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    stats = data.get('statistics', {})
    analyses = data.get('analyses', [])
    
    print("=" * 70)
    print("LLM ANALYSIS VALUE ASSESSMENT")
    print("=" * 70)
    
    print(f"\n## Overall Stats")
    print(f"Total analyzed: {stats.get('total_analyzed', 0)}")
    print(f"Success rate: {stats.get('successful', 0)}/{stats.get('total_analyzed', 0)} ({100*stats.get('successful',0)/max(1,stats.get('total_analyzed',1)):.1f}%)")
    
    # Topic diversity
    topics = stats.get('by_topic', {})
    print(f"\n## Topic Diversity: {len(topics)} unique topics identified")
    print("Top 15 topics:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {topic}: {count}")
    
    # Topic categories
    cats = stats.get('by_topic_category', {})
    print(f"\n## Topic Categories:")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Study types
    stypes = stats.get('study_types', {})
    print(f"\n## Study Types:")
    for stype, count in stypes.items():
        if count > 0:
            print(f"  {stype}: {count}")
    
    # MoTrPAC utility
    util = stats.get('motrpac_utility', {})
    print(f"\n## MoTrPAC Utility (KEY VALUE):")
    print(f"  Rat studies: {util.get('rat_studies', 0)}")
    print(f"  GeneCompass useful: {util.get('genecompass_useful', 0)}")
    print(f"  Deconvolution useful: {util.get('deconvolution_useful', 0)}")
    print(f"  GRN useful: {util.get('grn_useful', 0)}")
    
    tissues = util.get('tissues_covered', {})
    if tissues:
        print(f"\n  MoTrPAC tissues covered:")
        for tissue, count in sorted(tissues.items(), key=lambda x: x[1], reverse=True):
            print(f"    {tissue}: {count}")
    
    # Validation stats
    val = stats.get('validation', {})
    print(f"\n## Validation (catches extraction errors):")
    print(f"  Organism correct: {val.get('organism_correct_pct', 0):.1f}%")
    print(f"  Tissues correct: {val.get('tissues_correct_pct', 0):.1f}%")
    
    # Disease types
    diseases = stats.get('disease_types', {})
    if diseases:
        print(f"\n## Disease Models Found:")
        for disease, count in sorted(diseases.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {disease}: {count}")
    
    # Sample detailed analyses
    print("\n" + "=" * 70)
    print("SAMPLE ANALYSES (showing value of detailed extraction)")
    print("=" * 70)
    
    # Find interesting examples
    exercise_studies = [a for a in analyses if a.get('treatments', {}).get('has_exercise')]
    disease_studies = [a for a in analyses if a.get('disease_condition', {}).get('has_disease_model')]
    deconv_useful = [a for a in analyses if a.get('utility_for_motrpac', {}).get('deconvolution_useful')]
    time_series = [a for a in analyses if a.get('study_type', {}).get('is_time_series')]
    validation_issues = [a for a in analyses if not a.get('metadata_validation', {}).get('extracted_organism_correct') 
                        or not a.get('metadata_validation', {}).get('extracted_tissues_correct')]
    
    print(f"\n## Exercise studies found: {len(exercise_studies)}")
    for s in exercise_studies[:3]:
        print(f"  - {s['accession']}: {s.get('study_overview', {}).get('title', 'N/A')[:60]}...")
        ex = s.get('treatments', {})
        print(f"    Type: {ex.get('exercise_type')}, Protocol details extracted")
    
    print(f"\n## Disease models found: {len(disease_studies)}")
    for s in disease_studies[:5]:
        disease = s.get('disease_condition', {})
        print(f"  - {s['accession']}: {disease.get('disease_name', 'N/A')} ({disease.get('disease_type', 'N/A')})")
        print(f"    Induction: {disease.get('induction_method', 'N/A')}")
    
    print(f"\n## Deconvolution references found: {len(deconv_useful)}")
    for s in deconv_useful[:5]:
        tissues = [t.get('motrpac_match') for t in s.get('tissues', []) if t.get('motrpac_match')]
        print(f"  - {s['accession']}: {s.get('study_overview', {}).get('primary_topic', 'N/A')}")
        print(f"    MoTrPAC tissues: {tissues}")
    
    print(f"\n## Time series studies: {len(time_series)}")
    for s in time_series[:5]:
        design = s.get('experimental_design', {})
        print(f"  - {s['accession']}: {s.get('study_overview', {}).get('title', 'N/A')[:50]}...")
        print(f"    Time points: {design.get('time_points', [])[:5]}")
    
    print(f"\n## Validation caught issues: {len(validation_issues)}")
    for s in validation_issues[:5]:
        val = s.get('metadata_validation', {})
        if not val.get('extracted_organism_correct'):
            print(f"  - {s['accession']}: Organism mismatch")
            print(f"    Extracted: {val.get('extracted_organism', 'N/A')}")
            print(f"    Actual: {val.get('actual_organism', 'N/A')}")
        if not val.get('extracted_tissues_correct'):
            print(f"  - {s['accession']}: Tissue mismatch")
            print(f"    Actual tissues: {val.get('actual_tissues', [])}")
    
    # Cost analysis
    print("\n" + "=" * 70)
    print("COST vs VALUE ANALYSIS")
    print("=" * 70)
    
    total_input = sum(a.get('_meta', {}).get('input_tokens', 0) for a in analyses)
    total_output = sum(a.get('_meta', {}).get('output_tokens', 0) for a in analyses)
    
    # Sonnet pricing
    input_cost = total_input / 1000 * 0.003
    output_cost = total_output / 1000 * 0.015
    total_cost = input_cost + output_cost
    
    print(f"\nToken usage for {len(analyses)} studies:")
    print(f"  Input tokens: {total_input:,} (${input_cost:.2f})")
    print(f"  Output tokens: {total_output:,} (${output_cost:.2f})")
    print(f"  Total cost so far: ${total_cost:.2f}")
    print(f"  Cost per study: ${total_cost/max(1,len(analyses)):.4f}")
    
    # Extrapolate
    total_studies = 2670
    projected_cost = total_cost / len(analyses) * total_studies if analyses else 0
    print(f"\n  Projected total cost (2670 studies): ${projected_cost:.2f}")
    
    # Value assessment
    print(f"\n## VALUE DELIVERED:")
    print(f"  ✓ {len(topics)} unique research topics identified")
    print(f"  ✓ {util.get('genecompass_useful', 0)} studies flagged for GeneCompass training")
    print(f"  ✓ {util.get('deconvolution_useful', 0)} single-cell references for deconvolution")
    print(f"  ✓ {util.get('grn_useful', 0)} studies suitable for GRN inference")
    print(f"  ✓ {len(validation_issues)} extraction errors caught")
    print(f"  ✓ Detailed disease models, treatments, time points extracted")
    print(f"  ✓ Strain, sex, age information standardized")
    
    # Alternative cost
    print(f"\n## ALTERNATIVE: Manual curation")
    print(f"  At 5 min/study for a human curator: {len(analyses) * 5 / 60:.1f} hours")
    print(f"  At $30/hr: ${len(analyses) * 5 / 60 * 30:.2f}")
    print(f"  For all 2670 studies: {2670 * 5 / 60:.1f} hours = ${2670 * 5 / 60 * 30:.2f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_llm_output.py <path_to_json>")
        sys.exit(1)
    analyze_output(sys.argv[1])