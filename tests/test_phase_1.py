"""
Phase I Validation Test: Shock Propagation Sanity Check

This test validates that the manifold geometry correctly separates
different types of technological shocks.

Theory Reference (paper/main.tex):
- Definition 1.5: Diffusion operator K_d with exponential kernel
- Definition 1.6: Displacement dynamics dA/dt = K_d[I_t]
- Phase I objective: Test if O*NET geometry passes covariance identification

Success Criteria:
- Robotics shock should primarily affect manufacturing/physical occupations
- Software shock should primarily affect STEM/analytical occupations
- Cross-contamination (e.g., Software Dev high on Robotics shock) indicates broken geometry

Usage:
    cd /home/spencer/Research/task-space-model
    PYTHONPATH=src python tests/test_phase_1.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from task_space.manifold import OnetManifold
from task_space.dynamics import DynamicsEngine


# Occupation labels for readability
OCCUPATION_NAMES = {
    '15-1252.00': 'Software Developers',
    '51-4041.00': 'Machinists',
    '29-1141.00': 'Registered Nurses',
    '41-3031.00': 'Securities Sales Agents',
    '53-3032.00': 'Heavy Truck Drivers',
    '25-1011.00': 'Business Teachers',
    '47-2111.00': 'Electricians',
    '13-2011.00': 'Accountants',
    '35-2014.00': 'Cooks, Restaurant',
    '43-4051.00': 'Customer Service Reps',
}


def create_feature_based_shock(manifold, target_keywords, magnitude=1.0):
    """
    Create a shock vector based on occupation scores for specific features.

    The shock intensity for each occupation is proportional to how much
    that occupation uses the targeted work activities/abilities.

    Args:
        manifold: OnetManifold with loaded data
        target_keywords: List of substrings to match in element names
        magnitude: Base shock intensity

    Returns:
        shock_vector: Array of shape (n_occupations,)
    """
    n_tasks = manifold.n_tasks
    shock = np.zeros(n_tasks)

    # Find matching element indices
    matching_indices = []
    matching_names = []

    for keyword in target_keywords:
        matches = manifold.find_elements_by_name(keyword)
        for element_id, full_name in matches:
            idx = manifold.get_element_index(element_id)
            if idx is not None and idx not in matching_indices:
                matching_indices.append(idx)
                matching_names.append(full_name)

    if not matching_indices:
        print(f"  WARNING: No elements found matching {target_keywords}")
        return shock

    print(f"  Targeting {len(matching_indices)} elements:")
    for name in matching_names[:5]:
        print(f"    - {name}")
    if len(matching_names) > 5:
        print(f"    ... and {len(matching_names) - 5} more")

    # Shock intensity = sum of occupation scores on targeted features
    # This creates a shock profile where occupations that use these
    # activities more heavily receive higher shock intensity
    for occ_idx in range(n_tasks):
        total_exposure = 0
        for feat_idx in matching_indices:
            total_exposure += manifold.task_vectors[occ_idx, feat_idx]
        shock[occ_idx] = (total_exposure / len(matching_indices)) * magnitude

    return shock


def run_diffusion(engine, kernel, shock, n_steps=5):
    """
    Propagate shock through the manifold using diffusion dynamics.

    Implements Definition 1.6: A_{t+1} = A_t + K_d[I_t]
    """
    n_tasks = len(shock)
    A_t = np.zeros(n_tasks)
    C_t = np.ones(n_tasks)

    for _ in range(n_steps):
        A_t, C_t = engine.evolve(A_t, C_t, kernel, shock)

    return A_t, C_t


def print_ranked_occupations(manifold, A_t, title):
    """Print occupations ranked by displacement exposure."""
    print(f"\n{'-' * 50}")
    print(f"{title}")
    print(f"{'-' * 50}")

    sorted_indices = np.argsort(A_t)[::-1]

    for rank, idx in enumerate(sorted_indices, 1):
        soc_code = manifold.task_ids[idx]
        name = OCCUPATION_NAMES.get(soc_code, soc_code)
        exposure = A_t[idx]
        print(f"  {rank}. {name:<25} ({soc_code}): {exposure:.4f}")


def validate_results(manifold, A_robotics, A_software):
    """
    Validate that shock propagation matches economic intuition.

    Returns:
        (passed, messages): Tuple of (bool, list of result messages)
    """
    messages = []

    # Get rankings
    robotics_ranking = np.argsort(A_robotics)[::-1]
    software_ranking = np.argsort(A_software)[::-1]

    robotics_top3_codes = [manifold.task_ids[i] for i in robotics_ranking[:3]]
    software_top3_codes = [manifold.task_ids[i] for i in software_ranking[:3]]

    # Expected outcomes
    PHYSICAL_OCCUPATIONS = {'51-4041.00', '47-2111.00', '53-3032.00', '35-2014.00'}
    COGNITIVE_OCCUPATIONS = {'15-1252.00', '13-2011.00', '25-1011.00', '41-3031.00'}

    # Check 1: Physical occupations should rank higher on robotics shock
    robotics_physical_count = sum(1 for code in robotics_top3_codes if code in PHYSICAL_OCCUPATIONS)
    robotics_cognitive_count = sum(1 for code in robotics_top3_codes if code in COGNITIVE_OCCUPATIONS)

    if robotics_physical_count >= robotics_cognitive_count:
        messages.append(f"PASS: Robotics shock favors physical occupations ({robotics_physical_count}/3 in top 3)")
    else:
        messages.append(f"WARN: Robotics shock doesn't clearly favor physical occupations ({robotics_physical_count}/3)")

    # Check 2: Cognitive occupations should rank higher on software shock
    software_cognitive_count = sum(1 for code in software_top3_codes if code in COGNITIVE_OCCUPATIONS)
    software_physical_count = sum(1 for code in software_top3_codes if code in PHYSICAL_OCCUPATIONS)

    if software_cognitive_count >= software_physical_count:
        messages.append(f"PASS: Software shock favors cognitive occupations ({software_cognitive_count}/3 in top 3)")
    else:
        messages.append(f"WARN: Software shock doesn't clearly favor cognitive occupations ({software_cognitive_count}/3)")

    # Check 3: Software Developers should NOT be top on robotics
    if '15-1252.00' in robotics_top3_codes:
        messages.append("FAIL: Software Developers in top 3 for Robotics shock - geometry may be broken")
        passed = False
    else:
        messages.append("PASS: Software Developers not in top 3 for Robotics shock")
        passed = True

    # Check 4: Machinists should NOT be top on software
    if '51-4041.00' in software_top3_codes:
        messages.append("FAIL: Machinists in top 3 for Software shock - geometry may be broken")
        passed = False
    else:
        messages.append("PASS: Machinists not in top 3 for Software shock")

    return passed, messages


def main():
    print("=" * 60)
    print("PHASE I VALIDATION: Shock Propagation Sanity Check")
    print("=" * 60)
    print("\nTheory: Definition 1.5-1.6 (Diffusion Operator & Dynamics)")
    print("Test: Do O*NET-based task distances produce sensible shock propagation?")

    # Step A: Load O*NET data
    # NOTE: include_level=True provides better discrimination but is slow (~4000 API calls)
    # For quick testing, use include_level=False (importance only)
    # For production/research, cache Level data or use include_level=True
    #
    # NOTE: use_cache=False because we need occupation_data for find_elements_by_name().
    # The cache only stores task_vectors/task_ids/element_ids, not the raw element details.
    print("\n[Step A] Loading O*NET manifold...")
    include_level = False  # Set True for better accuracy (slower)
    if include_level:
        print("  Using combined Importance*Level scoring (slow but accurate)")
    else:
        print("  Using Importance-only scoring (fast)")
    try:
        manifold = OnetManifold()
        manifold.load_data(include_level=include_level, use_cache=False)
    except Exception as e:
        print(f"ERROR: Failed to load O*NET data: {e}")
        print("\nMake sure .env contains valid ONET_API_KEY")
        return False

    if manifold.n_tasks == 0:
        print("ERROR: No occupations loaded. Check API credentials.")
        return False

    print(f"  Loaded {manifold.n_tasks} occupations with {len(manifold.element_ids)} features")

    # Step B: Build diffusion kernel (this replaces k-NN graph)
    print("\n[Step B] Building diffusion kernel...")

    # Use sigma based on typical distances in the feature space
    # For O*NET scores (typically 1-5 scale), sigma=2 is reasonable
    engine = DynamicsEngine(sigma=2.0)
    kernel = engine.build_diffusion_kernel(manifold.task_vectors)
    print(f"  Kernel shape: {kernel.shape}")
    print(f"  Diffusion length scale (sigma): {engine.sigma}")

    # Step C: Create shock vectors
    print("\n[Step C] Creating shock vectors...")

    # Robotics shock: targets physical/manual work activities
    print("\n  ROBOTICS SHOCK:")
    ROBOTICS_TARGETS = [
        "Handling and Moving Objects",
        "Controlling Machines",
        "Operating Vehicles",
        "Repairing",
        "Manual Dexterity",
    ]
    shock_robotics = create_feature_based_shock(manifold, ROBOTICS_TARGETS, magnitude=1.0)

    # Software shock: targets cognitive/analytical work activities
    print("\n  SOFTWARE SHOCK:")
    SOFTWARE_TARGETS = [
        "Analyzing Data",
        "Processing Information",
        "Working with Computers",
        "Thinking Creatively",
        "Mathematical Reasoning",
    ]
    shock_software = create_feature_based_shock(manifold, SOFTWARE_TARGETS, magnitude=1.0)

    # Step D: Propagate shocks
    print("\n[Step D] Propagating shocks (5 time steps)...")
    A_robotics, _ = run_diffusion(engine, kernel, shock_robotics, n_steps=5)
    A_software, _ = run_diffusion(engine, kernel, shock_software, n_steps=5)

    # Step E: Print results
    print("\n[Step E] Displacement exposure rankings...")
    print_ranked_occupations(manifold, A_robotics, "ROBOTICS SHOCK - Displacement Exposure (A_t)")
    print_ranked_occupations(manifold, A_software, "SOFTWARE SHOCK - Displacement Exposure (A_t)")

    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    passed, messages = validate_results(manifold, A_robotics, A_software)

    for msg in messages:
        print(f"  {msg}")

    print("\n" + "-" * 60)
    if passed:
        print("PHASE I STATUS: PASSED")
        print("O*NET geometry produces sensible shock propagation patterns.")
    else:
        print("PHASE I STATUS: NEEDS INVESTIGATION")
        print("Some validation checks failed. Review manifold construction.")
    print("-" * 60)

    return passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
