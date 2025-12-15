"""
Task 4: Check O*NET to CPS Crosswalk Coverage

Mapping path:
O*NET-SOC (8-digit) → 6-digit SOC → Census 2010 → CPS OCC2010
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from task_space import build_dwa_occupation_measures

OUTPUT_DIR = Path("temp/mobility_feasibility/outputs")
DATA_DIR = Path("temp/mobility_feasibility/data/crosswalks")


def load_onet_occupations():
    """Load O*NET-SOC codes from our measures."""
    measures = build_dwa_occupation_measures()
    return measures.occupation_codes


def load_census_soc_crosswalk():
    """Load Census 2010 to SOC crosswalk."""
    df = pd.read_csv(DATA_DIR / "census_to_soc.csv")
    return df


def extract_soc6_from_onet(onet_code):
    """Extract 6-digit SOC from O*NET-SOC code."""
    # O*NET format: XX-XXXX.XX → SOC 6-digit: XX-XXXX
    return onet_code[:7]


def build_crosswalk(onet_codes, census_soc_df):
    """
    Build O*NET → CPS crosswalk.

    Returns DataFrame with mappings and coverage stats.
    """
    results = []

    # Build SOC → Census lookup
    # Note: SOC codes in Census have varying formats (some with spaces)
    census_soc_df["SOC_Clean"] = census_soc_df["SOC_Code"].str.strip().str.replace(" ", "")
    soc_to_census = census_soc_df.set_index("SOC_Clean")["Census_Code"].to_dict()

    for onet_code in onet_codes:
        soc6 = extract_soc6_from_onet(onet_code)

        # Try exact match
        census_code = soc_to_census.get(soc6)

        # If no exact match, try broader SOC (5-digit)
        if census_code is None:
            soc5 = soc6[:6]  # XX-XXX
            # Look for any SOC starting with this
            matches = [k for k in soc_to_census.keys() if k.startswith(soc5)]
            if len(matches) == 1:
                census_code = soc_to_census[matches[0]]
            elif len(matches) > 1:
                # Multiple matches - take first (will aggregate)
                census_code = soc_to_census[matches[0]]

        results.append({
            "onet_soc": onet_code,
            "soc_6digit": soc6,
            "census_2010_code": census_code,
            "matched": census_code is not None,
        })

    return pd.DataFrame(results)


def analyze_coverage(crosswalk_df, census_soc_df):
    """Analyze crosswalk coverage."""
    n_onet = len(crosswalk_df)
    n_matched = crosswalk_df["matched"].sum()
    n_unmatched = n_onet - n_matched

    # Unique Census codes mapped
    n_census_codes = crosswalk_df["census_2010_code"].nunique()

    # O*NET codes per Census code (aggregation)
    onet_per_census = crosswalk_df[crosswalk_df["matched"]].groupby("census_2010_code").size()

    # Unmatched occupations
    unmatched = crosswalk_df[~crosswalk_df["matched"]]

    return {
        "n_onet_total": n_onet,
        "n_matched": int(n_matched),
        "n_unmatched": int(n_unmatched),
        "match_rate": float(n_matched / n_onet),
        "n_census_codes_used": int(n_census_codes),
        "n_census_codes_total": len(census_soc_df),
        "aggregation_stats": {
            "mean_onet_per_census": float(onet_per_census.mean()),
            "max_onet_per_census": int(onet_per_census.max()),
            "single_onet_census_codes": int((onet_per_census == 1).sum()),
            "multiple_onet_census_codes": int((onet_per_census > 1).sum()),
        },
        "unmatched_soc_codes": unmatched["soc_6digit"].tolist()[:20],  # Sample
    }


def main():
    print("=" * 60)
    print("O*NET TO CPS CROSSWALK COVERAGE CHECK")
    print("=" * 60)

    # Load data
    print("\nLoading O*NET occupations...")
    onet_codes = load_onet_occupations()
    print(f"  O*NET-SOC codes: {len(onet_codes)}")

    print("\nLoading Census → SOC crosswalk...")
    census_soc_df = load_census_soc_crosswalk()
    print(f"  Census 2010 codes: {len(census_soc_df)}")

    # Build crosswalk
    print("\nBuilding O*NET → Census crosswalk...")
    crosswalk_df = build_crosswalk(onet_codes, census_soc_df)

    # Analyze
    coverage = analyze_coverage(crosswalk_df, census_soc_df)

    # Print results
    print("\n" + "=" * 60)
    print("COVERAGE RESULTS")
    print("=" * 60)
    print(f"\nO*NET occupations: {coverage['n_onet_total']}")
    print(f"Matched to Census: {coverage['n_matched']} ({coverage['match_rate']*100:.1f}%)")
    print(f"Unmatched: {coverage['n_unmatched']}")
    print(f"\nCensus 2010 codes used: {coverage['n_census_codes_used']} / {coverage['n_census_codes_total']}")

    print("\n--- Aggregation (Multiple O*NET → Single Census) ---")
    agg = coverage["aggregation_stats"]
    print(f"Mean O*NET per Census code: {agg['mean_onet_per_census']:.2f}")
    print(f"Max O*NET per Census code: {agg['max_onet_per_census']}")
    print(f"Census codes with single O*NET: {agg['single_onet_census_codes']}")
    print(f"Census codes with multiple O*NET: {agg['multiple_onet_census_codes']}")

    if coverage["unmatched_soc_codes"]:
        print(f"\nSample unmatched SOC codes: {coverage['unmatched_soc_codes'][:10]}")

    # Feasibility assessment
    print("\n" + "=" * 60)
    print("FEASIBILITY ASSESSMENT")
    print("=" * 60)

    if coverage["match_rate"] >= 0.90:
        feasibility = "GOOD"
        note = "High coverage. Minor O*NET detail lost to aggregation."
    elif coverage["match_rate"] >= 0.70:
        feasibility = "ADEQUATE"
        note = "Acceptable coverage. Some analysis at Census-code level."
    else:
        feasibility = "PROBLEMATIC"
        note = "Low coverage. May need alternative crosswalk or aggregation."

    print(f"\nCrosswalk Feasibility: {feasibility}")
    print(f"Note: {note}")

    # Save results
    output = {
        "coverage": coverage,
        "feasibility": feasibility,
        "recommendation": note,
        "mapping_path": "O*NET-SOC (8-digit) → 6-digit SOC → Census 2010 → CPS OCC2010",
    }

    output_path = OUTPUT_DIR / "crosswalk_coverage.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")

    # Save full crosswalk
    crosswalk_path = DATA_DIR / "onet_to_census_crosswalk.csv"
    crosswalk_df.to_csv(crosswalk_path, index=False)
    print(f"Full crosswalk saved to: {crosswalk_path}")


if __name__ == "__main__":
    main()
