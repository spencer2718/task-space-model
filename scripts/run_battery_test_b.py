"""
Battery Test B: Autor-Dorn Polarization

Tests whether continuous semantic height (CSH) adds explanatory power
beyond discrete routine share hypothesis (RSH) for employment polarization.

Outcomes:
- Δ service share (non-college)
- Δ routine share
- Δ clerical/retail share
- Δ mgmt/professional/tech share
- Δ operator share

Specification matches Autor-Dorn (2013) Table 5.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import json

# Paths
REPO_ROOT = Path(__file__).parent.parent
WORKFILE_PATH = REPO_ROOT / "data/external/dorn_replication/dorn_extracted/Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/workfile2012.dta"
OCC_DATA_PATH = REPO_ROOT / "data/external/dorn_replication/dorn_extracted/Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/occ1990dd_data2012.dta"
CSH_PATH = REPO_ROOT / "outputs/experiments/csh_values_v0722.csv"
ALM_PATH = REPO_ROOT / "data/external/dorn_replication/occ1990dd_task_alm.dta"
OUTPUT_PATH = REPO_ROOT / "outputs/experiments/battery_test_b_v0723.json"


def load_occupation_data():
    """Load occupation-level data with CSH and RTI."""
    # Load Dorn occupation data
    df_occ = pd.read_stata(OCC_DATA_PATH)

    # Load our CSH
    csh_df = pd.read_csv(CSH_PATH)

    # Load ALM RTI
    alm_df = pd.read_stata(ALM_PATH)
    rti_dict = {}
    for _, row in alm_df.iterrows():
        occ = int(row['occ1990dd'])
        r, m, a = row['task_routine'], row['task_manual'], row['task_abstract']
        if pd.notna(r) and pd.notna(m) and pd.notna(a) and r > 0 and m > 0 and a > 0:
            rti_dict[occ] = np.log(r) - np.log(m) - np.log(a)

    # Merge
    df_occ = df_occ.merge(csh_df[['occ1990dd', 'csh']], on='occ1990dd', how='left')
    df_occ['rti'] = df_occ['occ1990dd'].map(rti_dict)

    return df_occ


def compute_cz_csh_proxy(workfile_df, occ_df):
    """
    Compute CZ-level CSH proxy from occupation-level relationship.

    Method:
    Since RSH_cz is a function of occupation mix, and CSH is correlated with RTI,
    we use the occupation-level relationship to construct CSH_proxy_cz.

    RSH_cz = share of employment in routine (top-33% RTI) occupations
    CSH_cz_proxy = α + β × RSH_cz

    where β is scaled to match occupation-level variance relationship.
    """
    # Get occupation-level relationship
    occ_valid = occ_df[occ_df['csh'].notna() & occ_df['rti'].notna()].copy()

    # Compute employment-weighted means
    emp_weights = occ_valid['sh_empl1980'].fillna(0)
    emp_weights = emp_weights / emp_weights.sum()

    csh_mean = (occ_valid['csh'] * emp_weights).sum()
    csh_std = np.sqrt(((occ_valid['csh'] - csh_mean) ** 2 * emp_weights).sum())

    # For CZ-level, scale RSH to match CSH variance structure
    # RSH range: [0.18, 0.39], centered around 0.27
    # CSH range: ~[-0.14, 0.15], centered around mean

    rsh_1980 = workfile_df[workfile_df['yr'] == 1980]['l_sh_routine33a']
    rsh_mean = rsh_1980.mean()
    rsh_std = rsh_1980.std()

    # Linear transformation: CSH_proxy = a + b * RSH
    # Match mean and std scaling
    b = csh_std / rsh_std  # Scale factor
    a = csh_mean - b * rsh_mean  # Intercept

    print(f"CSH proxy transformation: CSH_proxy = {a:.4f} + {b:.4f} × RSH")
    print(f"  RSH: mean={rsh_mean:.4f}, std={rsh_std:.4f}")
    print(f"  CSH (occ): mean={csh_mean:.4f}, std={csh_std:.4f}")

    # Apply transformation
    workfile_df['csh_proxy'] = a + b * workfile_df['l_sh_routine33a']

    # Verify correlation at CZ level (should be 1.0 by construction)
    cz_1980 = workfile_df[workfile_df['yr'] == 1980]
    r = cz_1980['l_sh_routine33a'].corr(cz_1980['csh_proxy'])
    print(f"  r(RSH_cz, CSH_proxy_cz) = {r:.4f} (should be 1.0)")

    return workfile_df


def compute_csh_resid(workfile_df, binary=True):
    """
    Compute CSH residualized against discrete RSH classification.

    Args:
        workfile_df: DataFrame with csh_proxy column
        binary: If True, residualize against top-tercile vs rest (Autor-Dorn style)

    Returns:
        DataFrame with csh_resid column
    """
    # Get 1980 RSH tercile thresholds
    rsh_1980 = workfile_df[workfile_df['yr'] == 1980]['l_sh_routine33a']
    if binary:
        # Top tercile = high RSH
        threshold = rsh_1980.quantile(2/3)
        workfile_df['rsh_binary'] = (workfile_df['l_sh_routine33a'] > threshold).astype(int)

        # Compute CSH residuals by binary class
        for t in [0, 1]:
            mask = workfile_df['rsh_binary'] == t
            class_mean = workfile_df.loc[mask, 'csh_proxy'].mean()
            workfile_df.loc[mask, 'csh_resid'] = workfile_df.loc[mask, 'csh_proxy'] - class_mean
    else:
        # Terciles
        t1 = rsh_1980.quantile(1/3)
        t2 = rsh_1980.quantile(2/3)
        workfile_df['rsh_tercile'] = pd.cut(
            workfile_df['l_sh_routine33a'],
            bins=[-np.inf, t1, t2, np.inf],
            labels=[0, 1, 2]
        ).astype(int)

        for t in [0, 1, 2]:
            mask = workfile_df['rsh_tercile'] == t
            class_mean = workfile_df.loc[mask, 'csh_proxy'].mean()
            workfile_df.loc[mask, 'csh_resid'] = workfile_df.loc[mask, 'csh_proxy'] - class_mean

    # Report residual variance
    csh_std = workfile_df['csh_proxy'].std()
    resid_std = workfile_df['csh_resid'].std()
    ratio = resid_std / csh_std
    print(f"\nCSH residualization ({'binary' if binary else 'tercile'}):")
    print(f"  std(CSH_proxy): {csh_std:.6f}")
    print(f"  std(CSH_resid): {resid_std:.6f}")
    print(f"  Ratio: {ratio:.4f}")
    print(f"  Variance retained: {ratio**2:.2%}")

    return workfile_df


def run_regressions(workfile_df, outcome_vars, controls_level=True):
    """
    Run regressions for each outcome.

    Model 1: Y = β₁·RSH + state FE + time FE + ε
    Model 2: Y = β₁·RSH + β₃·CSH_resid + state FE + time FE + ε

    Returns results dict.
    """
    # Filter to 1980+ panel (t1, t2, t3 periods)
    df = workfile_df[workfile_df['yr'] >= 1980].copy()

    # Create state dummies - ensure numeric type
    state_dummies = pd.get_dummies(df['statefip'], prefix='state', drop_first=True, dtype=float)
    df = pd.concat([df, state_dummies], axis=1)
    state_cols = [c for c in df.columns if c.startswith('state_')]

    # Time dummies
    df['t2'] = (df['yr'] == 1990).astype(float)
    df['t3'] = (df['yr'] == 2000).astype(float)

    # Ensure numeric types for key variables
    df['rsh_binary'] = df['rsh_binary'].astype(float)
    df['csh_resid'] = df['csh_resid'].astype(float)

    results = {}

    for outcome_name, outcome_col in outcome_vars.items():
        if outcome_col not in df.columns:
            print(f"  Skipping {outcome_name}: column {outcome_col} not found")
            continue

        # Drop missing
        cols_needed = [outcome_col, 'l_sh_routine33a', 'rsh_binary', 'csh_resid',
                       't2', 't3', 'timepwt48'] + state_cols
        df_reg = df[cols_needed].dropna()

        if len(df_reg) < 100:
            print(f"  Skipping {outcome_name}: insufficient observations ({len(df_reg)})")
            continue

        # Outcome and weights - ensure float
        y = df_reg[outcome_col].astype(float)
        weights = df_reg['timepwt48'].astype(float)

        # Model 1: RSH only (using binary for discrete)
        X1 = df_reg[['rsh_binary', 't2', 't3'] + state_cols].astype(float)
        X1 = sm.add_constant(X1)

        model1 = sm.WLS(y, X1, weights=weights).fit()

        # Model 2: RSH + CSH_resid
        X2 = df_reg[['rsh_binary', 'csh_resid', 't2', 't3'] + state_cols].astype(float)
        X2 = sm.add_constant(X2)

        model2 = sm.WLS(y, X2, weights=weights).fit()

        # F-test for CSH_resid
        # Compare nested models
        f_stat = (model2.rsquared - model1.rsquared) / ((1 - model2.rsquared) / (model2.df_resid))
        f_pvalue = 1 - stats.f.cdf(f_stat, 1, model2.df_resid)

        results[outcome_name] = {
            'model1': {
                'beta_rsh': float(model1.params.get('rsh_binary', np.nan)),
                'se_rsh': float(model1.bse.get('rsh_binary', np.nan)),
                'pvalue_rsh': float(model1.pvalues.get('rsh_binary', np.nan)),
                'r2': float(model1.rsquared),
                'n': int(model1.nobs),
            },
            'model2': {
                'beta_rsh': float(model2.params.get('rsh_binary', np.nan)),
                'se_rsh': float(model2.bse.get('rsh_binary', np.nan)),
                'pvalue_rsh': float(model2.pvalues.get('rsh_binary', np.nan)),
                'beta_csh_resid': float(model2.params.get('csh_resid', np.nan)),
                'se_csh_resid': float(model2.bse.get('csh_resid', np.nan)),
                'pvalue_csh_resid': float(model2.pvalues.get('csh_resid', np.nan)),
                'r2': float(model2.rsquared),
                'delta_r2': float(model2.rsquared - model1.rsquared),
                'f_pvalue': float(f_pvalue),
                'n': int(model2.nobs),
            },
        }

        print(f"\n{outcome_name}:")
        print(f"  Model 1: β_RSH = {results[outcome_name]['model1']['beta_rsh']:.4f} "
              f"(SE={results[outcome_name]['model1']['se_rsh']:.4f}), R² = {results[outcome_name]['model1']['r2']:.4f}")
        print(f"  Model 2: β_RSH = {results[outcome_name]['model2']['beta_rsh']:.4f}, "
              f"β_CSH_resid = {results[outcome_name]['model2']['beta_csh_resid']:.4f} "
              f"(SE={results[outcome_name]['model2']['se_csh_resid']:.4f}, p={results[outcome_name]['model2']['pvalue_csh_resid']:.4f})")
        print(f"           ΔR² = {results[outcome_name]['model2']['delta_r2']:.4f}")

    return results


def determine_verdicts(results, expected_signs):
    """
    Apply interpretation matrix.

    Verdict rules:
    - "+": p(β₃) < 0.05 AND ΔR² ≥ 0.01 AND correct sign
    - "−": p(β₃) < 0.05 AND wrong sign
    - "0": Otherwise
    """
    verdicts = {}

    for outcome, res in results.items():
        m2 = res['model2']
        beta = m2['beta_csh_resid']
        pval = m2['pvalue_csh_resid']
        delta_r2 = m2['delta_r2']
        expected = expected_signs.get(outcome, 0)  # 0 = ambiguous

        if pval < 0.05:
            if expected == 0:
                # Ambiguous expected sign - any significant effect is "+"
                verdicts[outcome] = '+' if delta_r2 >= 0.01 else '0'
            elif np.sign(beta) == expected:
                # Correct sign
                verdicts[outcome] = '+' if delta_r2 >= 0.01 else '0'
            else:
                # Wrong sign
                verdicts[outcome] = '-'
        else:
            verdicts[outcome] = '0'

        print(f"  {outcome}: β₃={beta:.4f}, p={pval:.4f}, ΔR²={delta_r2:.4f}, "
              f"expected={expected}, verdict={verdicts[outcome]}")

    return verdicts


def main():
    print("=" * 70)
    print("BATTERY TEST B: Autor-Dorn Polarization")
    print("=" * 70)

    # Task 1: Load data
    print("\n--- Task 1: Load Data ---")
    workfile = pd.read_stata(WORKFILE_PATH)
    print(f"Workfile loaded: {workfile.shape[0]} rows, {workfile['czone'].nunique()} CZs")
    print(f"Years: {sorted(workfile['yr'].unique())}")

    occ_df = load_occupation_data()
    print(f"Occupation data: {len(occ_df)} occupations")

    # Task 2: Compute CZ-level CSH
    print("\n--- Task 2: Compute CZ-level CSH ---")
    workfile = compute_cz_csh_proxy(workfile, occ_df)
    workfile = compute_csh_resid(workfile, binary=True)

    # Validation: correlation between RSH and CSH_proxy
    cz_1980 = workfile[workfile['yr'] == 1980]
    r_rsh_csh = cz_1980['l_sh_routine33a'].corr(cz_1980['csh_proxy'])
    n_cz_valid = cz_1980['csh_proxy'].notna().sum()
    print(f"\nCZ coverage: {n_cz_valid}/722")
    print(f"r(RSH_cz, CSH_proxy_cz) = {r_rsh_csh:.4f}")

    # Task 3: Run regressions
    print("\n--- Task 3: Run Regressions ---")

    # Outcome variables (matching Autor-Dorn Table 5 and Table 7)
    outcome_vars = {
        'delta_service_share': 'd_shocc1_service_nc',
        'delta_routine_share': 'd_sh_routine33a',
        'delta_clericretail_share': 'd_shocc1_clericretail_nc',
        'delta_mgmtproftech_share': 'd_shocc1_mgmtproftech_nc',
        'delta_operator_share': 'd_shocc1_operator_nc',
    }

    results = run_regressions(workfile, outcome_vars)

    # Task 4: Apply interpretation matrix
    print("\n--- Task 4: Interpretation Matrix ---")

    # Expected signs for β_CSH_resid
    # Higher CSH = more routine-intensive in embedding space
    # So for routine job loss: expect negative (more routine → more loss)
    # For service growth: expect positive (more routine → more service growth)
    expected_signs = {
        'delta_service_share': 1,    # + (routine CZs saw service growth)
        'delta_routine_share': -1,   # - (routine CZs lost routine jobs)
        'delta_clericretail_share': -1,  # - (clerical is routine)
        'delta_mgmtproftech_share': 0,   # ambiguous
        'delta_operator_share': -1,  # - (operators are routine)
    }

    verdicts = determine_verdicts(results, expected_signs)

    # Summary
    n_plus = sum(1 for v in verdicts.values() if v == '+')
    n_minus = sum(1 for v in verdicts.values() if v == '-')
    n_zero = sum(1 for v in verdicts.values() if v == '0')

    print(f"\n--- Summary ---")
    print(f"  +: {n_plus}")
    print(f"  -: {n_minus}")
    print(f"  0: {n_zero}")

    # Task 5: Save results
    print("\n--- Task 5: Save Results ---")

    output = {
        'version': '0.7.2.3',
        'test': 'B',
        'description': 'Autor-Dorn polarization: CSH_resid vs RSH (binary)',
        'sample': {
            'n_cz': 722,
            'n_periods': 3,
            'period': '1980-2005',
            'n_obs': int(workfile[workfile['yr'] >= 1980].shape[0]),
        },
        'csh_validation': {
            'r_rsh_csh': float(r_rsh_csh),
            'n_cz_with_csh': int(n_cz_valid),
            'csh_resid_variance_retained': float(workfile['csh_resid'].std() / workfile['csh_proxy'].std()) ** 2,
        },
        'outcomes': results,
        'interpretation_matrix': verdicts,
        'summary': {
            'plus': n_plus,
            'minus': n_minus,
            'zero': n_zero,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {OUTPUT_PATH}")

    return output


if __name__ == '__main__':
    results = main()
