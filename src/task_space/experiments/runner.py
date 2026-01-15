"""
Generic experiment runner.

Executes experiments defined by YAML configs.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> dict:
    """
    Execute an experiment from configuration.

    Args:
        config: Experiment configuration

    Returns:
        Results dictionary
    """
    from ..domain import build_dwa_occupation_measures
    from ..data import load_oes_panel, compute_wage_comovement, onet_to_soc, get_dwa_titles
    from ..data.artifacts import get_embeddings, get_distance_matrix
    from ..similarity.kernel import build_kernel_matrix
    from ..similarity.overlap import compute_jaccard_overlap, compute_kernel_overlap, compute_normalized_overlap
    from ..validation.regression import run_validation_regression
    from ..validation.permutation import run_permutation_test, run_cross_validation

    results = {
        'config': config.to_dict(),
        'git_commit': _get_git_commit(),
        'timestamp': datetime.utcnow().isoformat(),
    }

    # 1. Load data
    print(f"[1/6] Loading data...")
    measures = build_dwa_occupation_measures(config.onet_path)

    oes_panel = load_oes_panel(list(config.oes_years), config.oes_path)
    comovement = compute_wage_comovement(oes_panel)

    # Build crosswalk dict
    crosswalk = {}
    for onet_code in measures.occupation_codes:
        soc = onet_to_soc(onet_code)
        if soc in comovement.occupation_codes:
            crosswalk[onet_code] = soc

    results['data'] = {
        'n_occupations': len(measures.occupation_codes),
        'n_activities': len(measures.activity_ids),
        'n_comovement_codes': len(comovement.occupation_codes),
        'crosswalk_coverage': len(crosswalk),
    }

    # 2. Compute similarity
    print(f"[2/6] Computing similarity ({config.similarity})...")
    if config.similarity == 'jaccard':
        similarity = compute_jaccard_overlap(measures.occupation_matrix)
        sigma = None
    else:
        # Get activity titles for embeddings
        dwa_titles = get_dwa_titles(config.onet_path)
        activity_titles = [dwa_titles.get(aid, aid) for aid in measures.activity_ids]

        embeddings = get_embeddings(activity_titles)
        dist_matrix = get_distance_matrix(embeddings)
        K, sigma = build_kernel_matrix(dist_matrix)

        if config.similarity == 'kernel':
            similarity = compute_kernel_overlap(measures.occupation_matrix, K)
        elif config.similarity == 'normalized_kernel':
            import warnings
            warnings.warn(
                "normalized_kernel is deprecated for distance applications per HC1. "
                "Use 'wasserstein' instead. See DISTANCE_GUIDE.md for details.",
                DeprecationWarning,
                stacklevel=2
            )
            similarity = compute_normalized_overlap(measures.occupation_matrix, K)
        else:
            raise ValueError(f"Unknown similarity: {config.similarity}")

    results['similarity'] = {
        'type': config.similarity,
        'sigma': sigma,
    }

    # 3. Compute shock exposure if specified
    if config.shock:
        print(f"[3/6] Computing shock exposure ({config.shock})...")
        from ..shocks.propagation import compute_exposure_from_shock

        prop_result = compute_exposure_from_shock(
            measures, measures.occupation_matrix,
            config.shock, config.shock_args,
            K, sigma,
        )
        results['shock'] = {
            'type': config.shock,
            'args': config.shock_args,
            'exposure_stats': {
                'min': float(prop_result.E.min()),
                'max': float(prop_result.E.max()),
                'mean': float(prop_result.E.mean()),
                'std': float(prop_result.E.std()),
            }
        }
    else:
        print(f"[3/6] No shock specified (validation-only mode)")

    # 4. Run regression
    print(f"[4/6] Running validation regression...")
    reg_result = run_validation_regression(
        similarity, comovement.comovement_matrix,
        measures.occupation_codes, comovement.occupation_codes,
        crosswalk, cluster_by=config.cluster_by,
    )
    results['regression'] = {
        'beta': float(reg_result.beta[1]),  # similarity coefficient
        'se': float(reg_result.se[1]),
        't': float(reg_result.t[1]),
        'p': float(reg_result.p[1]),
        'r2': float(reg_result.r2),
        'n_pairs': reg_result.n_pairs,
        'n_clusters': reg_result.n_clusters,
    }

    # 5. Robustness
    import numpy as np

    # Build matched pair arrays for permutation/CV tests
    # Filter to occupations in crosswalk
    valid_onet_codes = [c for c in measures.occupation_codes if c in crosswalk]
    onet_to_idx = {c: i for i, c in enumerate(measures.occupation_codes)}
    soc_to_idx = {c: i for i, c in enumerate(comovement.occupation_codes)}

    x_pairs = []
    y_pairs = []
    for i, onet_i in enumerate(valid_onet_codes):
        for j, onet_j in enumerate(valid_onet_codes):
            if i >= j:
                continue
            sim_i = onet_to_idx[onet_i]
            sim_j = onet_to_idx[onet_j]
            com_i = soc_to_idx[crosswalk[onet_i]]
            com_j = soc_to_idx[crosswalk[onet_j]]

            x_val = similarity[sim_i, sim_j]
            y_val = comovement.comovement_matrix[com_i, com_j]

            if not np.isnan(x_val) and not np.isnan(y_val):
                x_pairs.append(x_val)
                y_pairs.append(y_val)

    x_valid = np.array(x_pairs)
    y_valid = np.array(y_pairs)

    if config.run_permutation:
        print(f"[5/6] Running permutation test (n={config.n_permutations})...")
        perm_result = run_permutation_test(
            x_valid, y_valid,
            n_permutations=config.n_permutations, seed=config.seed,
        )
        results['permutation'] = asdict(perm_result)
    else:
        print(f"[5/6] Skipping permutation test")

    if config.run_cv:
        print(f"[6/6] Running cross-validation (k={config.n_folds})...")
        cv_result = run_cross_validation(
            x_valid, y_valid,
            n_folds=config.n_folds, seed=config.seed,
        )
        results['cross_validation'] = asdict(cv_result)
    else:
        print(f"[6/6] Skipping cross-validation")

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"{config.name}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"  R^2 = {results['regression']['r2']:.5f}")
    print(f"  t  = {results['regression']['t']:.2f}")

    return results


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'
