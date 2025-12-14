#!/usr/bin/env python3
"""
Rerun permutation tests with fixed implementation.

Usage:
    PYTHONPATH=src python tests/rerun_permutation_tests.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from task_space.domain import build_dwa_activity_domain, build_dwa_occupation_measures
from task_space.crosswalk import (
    build_onet_oes_crosswalk,
    load_oes_panel,
    compute_wage_comovement,
    onet_to_soc,
)
from task_space.comparison import (
    REPRESENTATION_NAMES,
    run_permutation_test,
    load_onet_structured_dimension,
)

OUTPUT_DIR = Path("outputs/phase2")
ONET_PATH = Path("data/onet/db_30_0_excel")
N_PERMUTATIONS = 1000


def main():
    print("Rerunning permutation tests with fixed implementation...")
    print("=" * 60)

    # Load base data
    print("\n[1/3] Loading data...")
    domain = build_dwa_activity_domain(ONET_PATH)
    measures = build_dwa_occupation_measures(ONET_PATH)

    # Load wage comovement
    oes_panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023])
    oes_codes = oes_panel["OCC_CODE"].unique().tolist()

    crosswalk_map = {}
    for onet_code in measures.occupation_codes:
        soc = onet_to_soc(onet_code)
        if soc in oes_codes:
            crosswalk_map[onet_code] = soc

    comovement = compute_wage_comovement(oes_panel, min_years=4)

    # Load similarity matrices
    print("\n[2/3] Loading similarity matrices...")
    sim_data = np.load(OUTPUT_DIR / "similarity_matrices.npz")
    similarity_matrices = {k: sim_data[k] for k in sim_data.files}

    # Load structured data for C4-C7 codes
    structured_data = {}
    for dim, rep_id in [('abilities', 'C4'), ('skills', 'C5'), ('knowledge', 'C6')]:
        _, occs, _ = load_onet_structured_dimension(ONET_PATH, dim)
        structured_data[rep_id] = occs

    # Combined (C7) uses intersection of all
    common_occs = set(structured_data['C4']) & set(structured_data['C5']) & set(structured_data['C6'])
    structured_data['C7'] = sorted(common_occs)

    # Build codes_map and crosswalk_maps
    codes_map = {}
    crosswalk_maps = {}

    for rep_id in ['D1', 'D2', 'D3', 'C1', 'C2', 'C3']:
        codes_map[rep_id] = measures.occupation_codes
        crosswalk_maps[rep_id] = crosswalk_map

    for rep_id in ['C4', 'C5', 'C6', 'C7']:
        codes_map[rep_id] = structured_data[rep_id]
        crosswalk_maps[rep_id] = {
            onet: onet_to_soc(onet)
            for onet in structured_data[rep_id]
            if onet_to_soc(onet) in comovement.occupation_codes
        }

    # Run permutation tests
    print(f"\n[3/3] Running permutation tests ({N_PERMUTATIONS} permutations)...")
    print(f"      This will take a few minutes...")

    perm_results = {}

    for rep_id in sorted(similarity_matrices.keys()):
        sim = similarity_matrices[rep_id]
        print(f"  {rep_id}...", end=" ", flush=True)

        try:
            result = run_permutation_test(
                similarity_matrix=sim,
                wage_comovement=comovement.comovement_matrix,
                sim_codes=codes_map[rep_id],
                comovement_codes=comovement.occupation_codes,
                crosswalk_map=crosswalk_maps[rep_id],
                rep_id=rep_id,
                n_permutations=N_PERMUTATIONS,
            )
            perm_results[rep_id] = result
            print(f"obs_t={result.observed_t:.2f}, perm_mean={result.perm_t_mean:.2f}, p={result.p_value:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save results
    with open(OUTPUT_DIR / "permutation_tests.json", 'w') as f:
        json.dump({k: {
            'rep_id': v.rep_id,
            'observed_t': v.observed_t,
            'perm_t_mean': v.perm_t_mean,
            'perm_t_std': v.perm_t_std,
            'perm_t_max': v.perm_t_max,
            'p_value': v.p_value,
            'n_permutations': v.n_permutations,
        } for k, v in perm_results.items()}, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/permutation_tests.json")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'ID':<5} {'Name':<20} {'obs_t':>8} {'perm_mean':>10} {'p':>8}")
    print(f"{'-'*5} {'-'*20} {'-'*8} {'-'*10} {'-'*8}")

    for rep_id in sorted(perm_results.keys(), key=lambda k: -perm_results[k].observed_t):
        res = perm_results[rep_id]
        name = REPRESENTATION_NAMES.get(rep_id, rep_id)
        print(f"{rep_id:<5} {name:<20} {res.observed_t:>8.2f} {res.perm_t_mean:>10.2f} {res.p_value:>8.4f}")


if __name__ == "__main__":
    main()
