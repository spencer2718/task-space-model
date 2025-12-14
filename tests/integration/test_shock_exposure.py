import pytest
import pandas as pd
from pathlib import Path


@pytest.mark.slow
def test_v1_shock_face_validity():
    """
    v1 shock should rank cognitive occupations higher than physical.

    Expected high exposure: Accountants, Software Developers, Lawyers
    Expected low exposure: Construction Laborers, Roofers, Helpers
    """
    from task_space import build_dwa_occupation_measures
    from task_space.data.onet import get_dwa_titles
    from task_space.data.artifacts import get_embeddings, get_distance_matrix
    from task_space.similarity.kernel import build_kernel_matrix
    from task_space.shocks.propagation import compute_exposure_from_shock

    onet_path = Path("data/onet/db_30_0_excel")
    measures = build_dwa_occupation_measures(onet_path)

    # Build kernel
    dwa_titles = get_dwa_titles(onet_path)
    activity_titles = [dwa_titles.get(aid, aid) for aid in measures.activity_ids]
    embeddings = get_embeddings(activity_titles)
    dist_matrix = get_distance_matrix(embeddings)
    K, sigma = build_kernel_matrix(dist_matrix)

    # Compute v1 exposure
    result = compute_exposure_from_shock(
        measures, measures.occupation_matrix,
        "capability_v1",
        {"onet_path": onet_path},
        K,
    )

    # Create exposure ranking
    exposure_df = pd.DataFrame({
        'occ_code': measures.occupation_codes,
        'exposure': result.E,
    }).sort_values('exposure', ascending=False)

    n_occ = len(exposure_df)
    top_quartile = set(exposure_df.head(n_occ // 4)['occ_code'])
    bottom_quartile = set(exposure_df.tail(n_occ // 4)['occ_code'])

    # Cognitive occupations should be in top quartile
    # 13-20xx: Accountants, 15-12xx: Software Dev, 23-10xx: Lawyers
    cognitive_prefixes = ['13-20', '15-12', '23-10']
    cognitive_found = 0
    for prefix in cognitive_prefixes:
        matching = [c for c in top_quartile if c.startswith(prefix)]
        if len(matching) > 0:
            cognitive_found += 1

    assert cognitive_found >= 2, f"At least 2 cognitive occupation types should be in top quartile, found {cognitive_found}"

    # Physical occupations should be in bottom quartile
    # 47-20xx: Construction, 47-21xx: Roofers, 47-30xx: Helpers
    physical_prefixes = ['47-20', '47-21', '47-30']
    physical_found = 0
    for prefix in physical_prefixes:
        matching = [c for c in bottom_quartile if c.startswith(prefix)]
        if len(matching) > 0:
            physical_found += 1

    assert physical_found >= 2, f"At least 2 physical occupation types should be in bottom quartile, found {physical_found}"
