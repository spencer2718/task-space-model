"""
Crosswalk diagnostics for retrospective battery.

Provides coverage analysis, loss categorization, and validation utilities
for the occ1990dd → O*NET-SOC crosswalk.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import json


@dataclass
class CoverageReport:
    """
    Coverage report for occ1990dd → O*NET crosswalk.

    Attributes:
        total_codes: Total occ1990dd codes
        mapped_codes: Codes successfully mapped
        unmapped_codes: Codes that failed to map
        unweighted_coverage: Fraction of codes mapped
        weighted_coverage: Employment-weighted coverage
        unmapped_by_category: Unmapped codes grouped by occupation type
        top_unmapped: Top unmapped codes by employment
        confidence_distribution: Counts by confidence tier
        gate_passed: Whether 80% employment-weighted threshold met
    """
    total_codes: int
    mapped_codes: int
    unmapped_codes: int
    unweighted_coverage: float
    weighted_coverage: float
    unmapped_by_category: Dict[str, List[int]]
    top_unmapped: List[Dict]
    confidence_distribution: Dict[str, int]
    gate_passed: bool


# Occupation type classifications for occ1990dd codes
# Based on Census 1990 major groups
OCC_CATEGORIES = {
    'managerial': range(1, 38),
    'professional': range(43, 200),
    'technical': range(203, 236),
    'sales': range(243, 286),
    'administrative': range(303, 390),
    'service': range(403, 470),
    'farm_forest_fish': range(473, 500),
    'precision_production': range(503, 700),
    'operators': range(703, 800),
    'transport': range(803, 860),
    'laborers': range(863, 890),
}


def categorize_occ1990dd(code: int) -> str:
    """
    Categorize an occ1990dd code by occupation type.

    Args:
        code: occ1990dd code

    Returns:
        Category name (e.g., 'operators', 'service')
    """
    for category, code_range in OCC_CATEGORIES.items():
        if code in code_range:
            return category
    return 'other'


def generate_coverage_report(
    crosswalk_path: Path,
    ipums_path: Optional[Path] = None,
    employment_path: Optional[Path] = None,
) -> CoverageReport:
    """
    Generate detailed coverage report for crosswalk.

    Args:
        crosswalk_path: Path to crosswalk CSV
        ipums_path: Optional path to IPUMS crosswalk (for occupation titles)
        employment_path: Optional path to Dorn employment data

    Returns:
        CoverageReport with detailed analysis
    """
    df = pd.read_csv(crosswalk_path)

    # Basic counts
    all_codes = df['occ1990dd'].unique()
    mapped_codes = df[df['onet_soc'].notna()]['occ1990dd'].unique()
    unmapped_codes = df[df['onet_soc'].isna()]['occ1990dd'].unique()

    total = len(all_codes)
    n_mapped = len(mapped_codes)
    n_unmapped = len(unmapped_codes)

    unweighted_coverage = n_mapped / total if total > 0 else 0

    # Employment-weighted coverage
    emp_data = {}
    if employment_path and employment_path.exists():
        emp_df = pd.read_stata(employment_path)
        for _, row in emp_df.iterrows():
            code = int(row['occ1990dd'])
            emp = row.get('sh_empl1980', 0)
            if pd.notna(emp):
                emp_data[code] = emp

    if emp_data:
        total_emp = sum(emp_data.values())
        mapped_emp = sum(emp_data.get(c, 0) for c in mapped_codes)
        weighted_coverage = mapped_emp / total_emp if total_emp > 0 else unweighted_coverage
    else:
        weighted_coverage = unweighted_coverage

    # Categorize unmapped codes
    unmapped_by_category: Dict[str, List[int]] = {}
    for code in unmapped_codes:
        category = categorize_occ1990dd(code)
        if category not in unmapped_by_category:
            unmapped_by_category[category] = []
        unmapped_by_category[category].append(int(code))

    # Get occupation titles if IPUMS available
    titles = {}
    if ipums_path and ipums_path.exists():
        ipums = pd.read_excel(ipums_path)
        ipums_clean = ipums[ipums['1990 Census code'].astype(str).str.match(r'^\d+$', na=False)].copy()
        ipums_clean['census_1990'] = ipums_clean['1990 Census code'].astype(int)
        for _, row in ipums_clean.iterrows():
            titles[row['census_1990']] = row['1990 Census title']

    # Top unmapped by employment
    top_unmapped = []
    for code in unmapped_codes:
        emp = emp_data.get(code, 0)
        title = titles.get(code, f"Unknown (code {code})")
        category = categorize_occ1990dd(code)
        top_unmapped.append({
            'occ1990dd': int(code),
            'title': title,
            'category': category,
            'emp_1980': float(emp),
        })

    top_unmapped.sort(key=lambda x: x['emp_1980'], reverse=True)

    # Confidence tier distribution
    tier_counts = df[df['onet_soc'].notna()]['confidence_tier'].value_counts().to_dict()
    tier_counts['unmapped'] = n_unmapped

    return CoverageReport(
        total_codes=total,
        mapped_codes=n_mapped,
        unmapped_codes=n_unmapped,
        unweighted_coverage=unweighted_coverage,
        weighted_coverage=weighted_coverage,
        unmapped_by_category=unmapped_by_category,
        top_unmapped=top_unmapped[:20],  # Top 20
        confidence_distribution=tier_counts,
        gate_passed=weighted_coverage >= 0.80,
    )


def print_coverage_report(report: CoverageReport) -> None:
    """Print formatted coverage report to console."""
    print("=" * 70)
    print("OCC1990DD → O*NET-SOC CROSSWALK COVERAGE REPORT")
    print("=" * 70)
    print()

    print("SUMMARY")
    print("-" * 40)
    print(f"Total occ1990dd codes:       {report.total_codes}")
    print(f"Mapped codes:                {report.mapped_codes} ({report.unweighted_coverage*100:.1f}%)")
    print(f"Unmapped codes:              {report.unmapped_codes}")
    print()
    print(f"Unweighted coverage:         {report.unweighted_coverage*100:.1f}%")
    print(f"Employment-weighted coverage: {report.weighted_coverage*100:.1f}%")
    print()
    print(f"80% Gate: {'PASSED ✓' if report.gate_passed else 'FAILED ✗'}")
    print()

    print("CONFIDENCE TIER DISTRIBUTION")
    print("-" * 40)
    for tier, count in sorted(report.confidence_distribution.items()):
        print(f"  {tier:<12}: {count}")
    print()

    print("UNMAPPED CODES BY CATEGORY")
    print("-" * 40)
    for category, codes in sorted(report.unmapped_by_category.items(),
                                   key=lambda x: len(x[1]), reverse=True):
        print(f"  {category:<20}: {len(codes)} codes")
    print()

    print("TOP UNMAPPED BY 1980 EMPLOYMENT")
    print("-" * 40)
    for item in report.top_unmapped[:10]:
        print(f"  {item['occ1990dd']:4d}: {item['title'][:45]:<45} ({item['emp_1980']*100:.2f}%)")
    print()


def save_coverage_report(report: CoverageReport, output_path: Path) -> None:
    """
    Save coverage report to JSON file.

    Args:
        report: CoverageReport to save
        output_path: Path for output JSON
    """
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    data = {
        'total_codes': int(report.total_codes),
        'mapped_codes': int(report.mapped_codes),
        'unmapped_codes': int(report.unmapped_codes),
        'unweighted_coverage': float(report.unweighted_coverage),
        'weighted_coverage': float(report.weighted_coverage),
        'unmapped_by_category': convert_types(report.unmapped_by_category),
        'top_unmapped': convert_types(report.top_unmapped),
        'confidence_distribution': convert_types(report.confidence_distribution),
        'gate_passed': bool(report.gate_passed),
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_crosswalk(crosswalk_path: Path) -> Dict[str, any]:
    """
    Validate crosswalk file structure and completeness.

    Args:
        crosswalk_path: Path to crosswalk CSV

    Returns:
        Dict with validation results
    """
    df = pd.read_csv(crosswalk_path)

    required_cols = ['occ1990dd', 'onet_soc', 'weight', 'confidence_tier', 'emp_1980']
    missing_cols = [c for c in required_cols if c not in df.columns]

    # Check weights sum to 1 for each mapped occ1990dd
    weight_issues = []
    for occ, group in df[df['onet_soc'].notna()].groupby('occ1990dd'):
        total_weight = group['weight'].sum()
        if abs(total_weight - 1.0) > 0.01:
            weight_issues.append({
                'occ1990dd': int(occ),
                'total_weight': float(total_weight),
            })

    # Check for duplicate rows
    duplicates = df[df.duplicated(subset=['occ1990dd', 'onet_soc'], keep=False)]

    return {
        'valid': len(missing_cols) == 0 and len(weight_issues) == 0,
        'missing_columns': missing_cols,
        'weight_issues': weight_issues[:10],  # First 10
        'duplicate_count': len(duplicates),
        'n_rows': len(df),
        'n_unique_occ1990dd': df['occ1990dd'].nunique(),
        'n_unique_onet': df['onet_soc'].nunique(),
    }
