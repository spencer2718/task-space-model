#!/usr/bin/env python3
"""
Phase 2: Systematic Representation Comparison (v0.6.2)

This script implements the full Phase 2 comparison protocol from spec_0.6.2.md.
Tests discrete, continuous (text embedding), continuous (O*NET structured),
and hybrid representations of task space.

Usage:
    PYTHONPATH=src python tests/run_phase2_comparison.py

Expected runtime:
    - Embedding computation: ~5-10 min (JobBERT slower than MPNet)
    - Permutation tests: ~10-15 min (1000 perms × representations)
    - Total: ~20-30 min
"""

import sys
from pathlib import Path

# Add src to path
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
    # Discrete
    compute_binary_jaccard,
    compute_weighted_jaccard,
    compute_cosine_binary,
    # Text embeddings
    compute_mpnet_embeddings,
    compute_jobbert_embeddings,
    compute_e5_embeddings,
    embeddings_to_kernel_overlap,
    # O*NET structured
    load_onet_structured_dimension,
    compute_structured_similarity,
    # Validation
    run_validation_regression,
    run_permutation_test,
    run_cross_validation,
    run_hybrid_regression,
    # Output
    save_phase2_results,
    RepresentationResult,
)


# Configuration
ONET_PATH = Path("data/onet/db_30_0_excel")
OUTPUT_DIR = Path("outputs/phase2")
DEVICE = "cuda"  # or "cpu"
N_PERMUTATIONS = 1000
N_CV_FOLDS = 5


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 2: Systematic Representation Comparison (v0.6.2)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load Base Data
    # =========================================================================
    print("\n[1/8] Loading base data...")

    # Build DWA domain and occupation measures
    domain = build_dwa_activity_domain(ONET_PATH)
    measures = build_dwa_occupation_measures(ONET_PATH)

    activity_titles = [domain.activity_names[aid] for aid in domain.activity_ids]

    print(f"  - {domain.n_activities} activities (DWAs)")
    print(f"  - {len(measures.occupation_codes)} occupations")

    # Load OES wage data
    print("\n  Loading wage comovement data...")
    oes_panel = load_oes_panel(years=[2019, 2020, 2021, 2022, 2023])
    oes_codes = oes_panel["OCC_CODE"].unique().tolist()
    print(f"  - Loaded OES data for {len(oes_codes)} SOC codes")

    # Build crosswalk
    crosswalk = build_onet_oes_crosswalk(
        onet_codes=measures.occupation_codes,
        oes_codes=oes_codes,
    )
    print(f"  - Crosswalk coverage: {crosswalk.coverage:.1%}")

    # Build crosswalk map
    crosswalk_map = {}
    for onet_code in measures.occupation_codes:
        soc = onet_to_soc(onet_code)
        if soc in oes_codes:
            crosswalk_map[onet_code] = soc

    # Compute wage comovement
    comovement = compute_wage_comovement(oes_panel, min_years=4)
    print(f"  - {len(comovement.occupation_codes)} SOC occupations with sufficient data")

    # =========================================================================
    # Step 2: Compute Discrete Representations
    # =========================================================================
    print("\n[2/8] Computing discrete representations...")

    similarity_matrices = {}
    sigmas = {}  # For continuous representations

    # D1: Binary Jaccard
    print("  D1: Binary Jaccard...")
    similarity_matrices['D1'] = compute_binary_jaccard(measures.raw_matrix)

    # D2: Weighted Jaccard
    print("  D2: Weighted Jaccard...")
    similarity_matrices['D2'] = compute_weighted_jaccard(measures.raw_matrix)

    # D3: Cosine (binary)
    print("  D3: Cosine (binary)...")
    similarity_matrices['D3'] = compute_cosine_binary(measures.raw_matrix)

    # =========================================================================
    # Step 3: Compute Text Embedding Representations
    # =========================================================================
    print("\n[3/8] Computing text embedding representations...")

    # Try to load existing embeddings first
    embeddings_file = OUTPUT_DIR / "activity_embeddings_all.npz"

    if embeddings_file.exists():
        print("  Loading cached embeddings...")
        cached = np.load(embeddings_file)
        mpnet_emb = cached['mpnet']
        try:
            jobbert_emb = cached['jobbert']
        except KeyError:
            jobbert_emb = None
        try:
            e5_emb = cached['e5']
        except KeyError:
            e5_emb = None
    else:
        mpnet_emb = None
        jobbert_emb = None
        e5_emb = None

    # C1: MPNet
    if mpnet_emb is None:
        print("  C1: Computing MPNet embeddings...")
        mpnet_emb = compute_mpnet_embeddings(activity_titles, device=DEVICE)
    else:
        print("  C1: Using cached MPNet embeddings")

    overlap_c1, sigma_c1 = embeddings_to_kernel_overlap(
        measures.occupation_matrix, mpnet_emb, normalize=False
    )
    similarity_matrices['C1'] = overlap_c1
    sigmas['C1'] = sigma_c1
    print(f"      σ = {sigma_c1:.4f}")

    # C2: JobBERT
    try:
        if jobbert_emb is None:
            print("  C2: Computing JobBERT embeddings...")
            jobbert_emb = compute_jobbert_embeddings(activity_titles, device=DEVICE)
        else:
            print("  C2: Using cached JobBERT embeddings")

        overlap_c2, sigma_c2 = embeddings_to_kernel_overlap(
            measures.occupation_matrix, jobbert_emb, normalize=False
        )
        similarity_matrices['C2'] = overlap_c2
        sigmas['C2'] = sigma_c2
        print(f"      σ = {sigma_c2:.4f}")
    except Exception as e:
        print(f"  C2: JobBERT SKIPPED ({e})")
        similarity_matrices['C2'] = None

    # C3: E5-large
    try:
        if e5_emb is None:
            print("  C3: Computing E5-large embeddings...")
            e5_emb = compute_e5_embeddings(activity_titles, device=DEVICE)
        else:
            print("  C3: Using cached E5-large embeddings")

        overlap_c3, sigma_c3 = embeddings_to_kernel_overlap(
            measures.occupation_matrix, e5_emb, normalize=False
        )
        similarity_matrices['C3'] = overlap_c3
        sigmas['C3'] = sigma_c3
        print(f"      σ = {sigma_c3:.4f}")
    except Exception as e:
        print(f"  C3: E5-large SKIPPED ({e})")
        similarity_matrices['C3'] = None

    # Save embeddings for future use
    emb_to_save = {'mpnet': mpnet_emb}
    if jobbert_emb is not None:
        emb_to_save['jobbert'] = jobbert_emb
    if e5_emb is not None:
        emb_to_save['e5'] = e5_emb
    np.savez_compressed(embeddings_file, **emb_to_save)

    # =========================================================================
    # Step 4: Compute O*NET Structured Representations
    # =========================================================================
    print("\n[4/8] Computing O*NET structured representations...")

    # These are at the occupation level, not activity level
    # Need separate handling

    structured_data = {}

    # C4: Abilities
    print("  C4: Abilities...")
    abilities, ability_occs, _ = load_onet_structured_dimension(ONET_PATH, 'abilities')
    structured_data['C4'] = {
        'vectors': abilities,
        'codes': ability_occs,
    }
    similarity_matrices['C4'] = compute_structured_similarity(abilities)
    print(f"      {abilities.shape[1]} dimensions, {len(ability_occs)} occupations")

    # C5: Skills
    print("  C5: Skills...")
    skills, skill_occs, _ = load_onet_structured_dimension(ONET_PATH, 'skills')
    structured_data['C5'] = {
        'vectors': skills,
        'codes': skill_occs,
    }
    similarity_matrices['C5'] = compute_structured_similarity(skills)
    print(f"      {skills.shape[1]} dimensions, {len(skill_occs)} occupations")

    # C6: Knowledge
    print("  C6: Knowledge...")
    knowledge, know_occs, _ = load_onet_structured_dimension(ONET_PATH, 'knowledge')
    structured_data['C6'] = {
        'vectors': knowledge,
        'codes': know_occs,
    }
    similarity_matrices['C6'] = compute_structured_similarity(knowledge)
    print(f"      {knowledge.shape[1]} dimensions, {len(know_occs)} occupations")

    # C7: Combined
    print("  C7: Combined (Abilities + Skills + Knowledge)...")
    # Need to align occupation codes across dimensions
    common_occs = set(ability_occs) & set(skill_occs) & set(know_occs)
    common_occs = sorted(common_occs)

    idx_a = [ability_occs.index(o) for o in common_occs]
    idx_s = [skill_occs.index(o) for o in common_occs]
    idx_k = [know_occs.index(o) for o in common_occs]

    combined = np.hstack([
        abilities[idx_a],
        skills[idx_s],
        knowledge[idx_k],
    ])
    structured_data['C7'] = {
        'vectors': combined,
        'codes': common_occs,
    }
    similarity_matrices['C7'] = compute_structured_similarity(combined)
    print(f"      {combined.shape[1]} dimensions, {len(common_occs)} occupations")

    # =========================================================================
    # Step 5: Primary Validation
    # =========================================================================
    print("\n[5/8] Running primary validation...")

    primary_results = {}

    # Map representation to its codes and crosswalk
    codes_map = {}
    crosswalk_maps = {}

    # Discrete and text embeddings use DWA occupation codes
    for rep_id in ['D1', 'D2', 'D3', 'C1', 'C2', 'C3']:
        codes_map[rep_id] = measures.occupation_codes
        crosswalk_maps[rep_id] = crosswalk_map

    # O*NET structured use their own occupation codes
    for rep_id in ['C4', 'C5', 'C6', 'C7']:
        codes_map[rep_id] = structured_data[rep_id]['codes']
        # These are already O*NET-SOC codes, need crosswalk
        crosswalk_maps[rep_id] = {
            onet: onet_to_soc(onet)
            for onet in structured_data[rep_id]['codes']
            if onet_to_soc(onet) in comovement.occupation_codes
        }

    # Categories
    categories = {
        'D1': 'discrete', 'D2': 'discrete', 'D3': 'discrete',
        'C1': 'continuous_text', 'C2': 'continuous_text', 'C3': 'continuous_text',
        'C4': 'continuous_onet', 'C5': 'continuous_onet',
        'C6': 'continuous_onet', 'C7': 'continuous_onet',
    }

    print(f"\n  {'ID':<5} {'Name':<20} {'t-stat':>10} {'R²':>12} {'n_pairs':>10}")
    print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*12} {'-'*10}")

    for rep_id in ['D1', 'D2', 'D3', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']:
        sim = similarity_matrices.get(rep_id)
        if sim is None:
            print(f"  {rep_id:<5} SKIPPED (no data)")
            continue

        try:
            result = run_validation_regression(
                similarity_matrix=sim,
                wage_comovement=comovement.comovement_matrix,
                sim_codes=codes_map[rep_id],
                comovement_codes=comovement.occupation_codes,
                crosswalk_map=crosswalk_maps[rep_id],
                rep_id=rep_id,
                name=REPRESENTATION_NAMES[rep_id],
                category=categories[rep_id],
                sigma=sigmas.get(rep_id),
            )
            primary_results[rep_id] = result
            print(f"  {rep_id:<5} {result.name:<20} {result.t_stat:>10.2f} {result.r_squared:>12.6f} {result.n_pairs:>10}")
        except Exception as e:
            print(f"  {rep_id:<5} ERROR: {e}")

    # Save intermediate results
    with open(OUTPUT_DIR / "primary_validation.json", 'w') as f:
        json.dump({k: {
            'rep_id': v.rep_id,
            'name': v.name,
            'category': v.category,
            'beta': v.beta,
            'se': v.se,
            't_stat': v.t_stat,
            'pvalue': v.pvalue,
            'r_squared': v.r_squared,
            'n_pairs': v.n_pairs,
            'n_clusters': v.n_clusters,
            'sigma': v.sigma,
        } for k, v in primary_results.items()}, f, indent=2)

    # =========================================================================
    # Step 6: Permutation Tests
    # =========================================================================
    print(f"\n[6/8] Running permutation tests ({N_PERMUTATIONS} permutations)...")

    perm_results = {}

    for rep_id in primary_results.keys():
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
            print(f"p = {result.p_value:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save intermediate
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

    # =========================================================================
    # Step 7: Cross-Validation
    # =========================================================================
    print(f"\n[7/8] Running cross-validation ({N_CV_FOLDS}-fold)...")

    cv_results = {}

    for rep_id in primary_results.keys():
        sim = similarity_matrices[rep_id]
        print(f"  {rep_id}...", end=" ", flush=True)

        try:
            result = run_cross_validation(
                similarity_matrix=sim,
                wage_comovement=comovement.comovement_matrix,
                sim_codes=codes_map[rep_id],
                comovement_codes=comovement.occupation_codes,
                crosswalk_map=crosswalk_maps[rep_id],
                rep_id=rep_id,
                n_folds=N_CV_FOLDS,
            )
            cv_results[rep_id] = result
            print(f"CV R² = {result.cv_r2_mean:.6f}, overfit = {result.overfit_ratio:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save intermediate
    with open(OUTPUT_DIR / "cross_validation.json", 'w') as f:
        json.dump({k: {
            'rep_id': v.rep_id,
            'cv_r2_mean': v.cv_r2_mean,
            'cv_r2_std': v.cv_r2_std,
            'cv_r2_folds': v.cv_r2_folds,
            'full_sample_r2': v.full_sample_r2,
            'overfit_ratio': v.overfit_ratio,
        } for k, v in cv_results.items()}, f, indent=2)

    # =========================================================================
    # Step 8: Hybrid Models
    # =========================================================================
    print("\n[8/8] Running hybrid models...")

    hybrid_results = {}

    # H1: Jaccard + MPNet
    if 'D1' in primary_results and 'C1' in primary_results:
        print("  H1: D1 + C1 (Jaccard + MPNet)...", end=" ", flush=True)
        try:
            h1 = run_hybrid_regression(
                similarity_matrices={'D1': similarity_matrices['D1'], 'C1': similarity_matrices['C1']},
                wage_comovement=comovement.comovement_matrix,
                codes_map={'D1': codes_map['D1'], 'C1': codes_map['C1']},
                comovement_codes=comovement.occupation_codes,
                crosswalk_maps={'D1': crosswalk_maps['D1'], 'C1': crosswalk_maps['C1']},
                components=['D1', 'C1'],
                model_id='H1',
            )
            hybrid_results['H1'] = h1
            print(f"R² = {h1.r_squared:.6f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # H2: Jaccard + Abilities
    if 'D1' in primary_results and 'C4' in primary_results:
        print("  H2: D1 + C4 (Jaccard + Abilities)...", end=" ", flush=True)
        try:
            h2 = run_hybrid_regression(
                similarity_matrices={'D1': similarity_matrices['D1'], 'C4': similarity_matrices['C4']},
                wage_comovement=comovement.comovement_matrix,
                codes_map={'D1': codes_map['D1'], 'C4': codes_map['C4']},
                comovement_codes=comovement.occupation_codes,
                crosswalk_maps={'D1': crosswalk_maps['D1'], 'C4': crosswalk_maps['C4']},
                components=['D1', 'C4'],
                model_id='H2',
            )
            hybrid_results['H2'] = h2
            print(f"R² = {h2.r_squared:.6f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Save hybrid results
    with open(OUTPUT_DIR / "hybrid_models.json", 'w') as f:
        json.dump({k: {
            'model_id': v.model_id,
            'components': v.components,
            'r_squared': v.r_squared,
            'n_pairs': v.n_pairs,
            'component_betas': v.component_betas,
            'component_ses': v.component_ses,
            'component_ts': v.component_ts,
            'component_ps': v.component_ps,
        } for k, v in hybrid_results.items()}, f, indent=2)

    # =========================================================================
    # Save All Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving results...")

    # Filter out None similarity matrices
    valid_sims = {k: v for k, v in similarity_matrices.items() if v is not None}
    save_phase2_results(
        output_dir=OUTPUT_DIR,
        primary_results=primary_results,
        perm_results=perm_results,
        cv_results=cv_results,
        hybrid_results=hybrid_results,
        similarity_matrices=valid_sims,
    )

    # =========================================================================
    # Print Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    # Find best representation
    if primary_results:
        best_id = max(primary_results.keys(), key=lambda k: primary_results[k].t_stat)
        best = primary_results[best_id]

        print(f"\nBest representation: {best_id} ({best.name})")
        print(f"  t-stat: {best.t_stat:.2f}")
        print(f"  R²: {best.r_squared:.6f}")

        if best_id in perm_results:
            print(f"  Permutation p: {perm_results[best_id].p_value:.4f}")

        if best_id in cv_results:
            print(f"  CV R²: {cv_results[best_id].cv_r2_mean:.6f}")
            print(f"  Overfit ratio: {cv_results[best_id].overfit_ratio:.2f}")

    # Tier classification
    print("\n  Tier Classification:")
    for rep_id in sorted(primary_results.keys(), key=lambda k: -primary_results[k].t_stat):
        res = primary_results[rep_id]
        perm_p = perm_results[rep_id].p_value if rep_id in perm_results else 1.0
        overfit = cv_results[rep_id].overfit_ratio if rep_id in cv_results else float('inf')

        if res.t_stat > 20 and perm_p < 0.001 and overfit < 1.2:
            tier = "Tier 1 (Recommended)"
        elif res.t_stat > 10 and perm_p < 0.01 and overfit < 1.5:
            tier = "Tier 2 (Acceptable)"
        elif res.t_stat > 5 and perm_p < 0.05:
            tier = "Tier 3 (Marginal)"
        else:
            tier = "Reject"

        print(f"    {rep_id}: {tier}")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - phase2_summary.md")
    print(f"  - primary_validation.json")
    print(f"  - permutation_tests.json")
    print(f"  - cross_validation.json")
    print(f"  - hybrid_models.json")
    print(f"  - similarity_matrices.npz")


if __name__ == "__main__":
    main()
