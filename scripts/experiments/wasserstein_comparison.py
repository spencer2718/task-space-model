"""
Wasserstein vs Kernel Overlap Comparison (Path A)

Pre-registered test: Does Wasserstein distance outperform kernel overlap
for predicting CPS worker mobility?

Pre-committed interpretations:
- OT >> Kernel: Metric structure matters; adopt OT
- Kernel >> OT: Ground metric noise dominates; stay with kernel
- OT ≈ Kernel: Both capture same information; prefer kernel (simpler)

Usage:
    python scripts/experiments/wasserstein_comparison.py

Output:
    outputs/experiments/wasserstein_comparison_v070.json
"""
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_DIR = Path("outputs/experiments")
VERSION = "0.7.0"


def main():
    """Run Wasserstein vs Kernel comparison experiment."""
    print("=" * 60)
    print("Wasserstein vs Kernel Overlap Comparison (Path A)")
    print("=" * 60)
    print()
    print("This experiment is a stub. Full implementation requires:")
    print("  1. Computing full 894×894 Wasserstein distance matrix (~30-60 min)")
    print("  2. Aggregating to Census occupation codes for CPS comparison")
    print("  3. Running conditional logit with d_wasserstein vs d_kernel")
    print("  4. Comparing log-likelihood and coefficient estimates")
    print()
    print("Run after verifying basic module functionality with:")
    print("  pytest tests/unit/similarity/test_wasserstein.py -v")
    print("  pytest tests/integration/similarity/ -v -s -m slow")
    print()

    # Placeholder for results structure
    results = {
        "experiment": "wasserstein_comparison",
        "version": VERSION,
        "status": "stub",
        "timestamp": datetime.now().isoformat(),
        "pre_committed_interpretations": {
            "OT >> Kernel": "Metric structure matters; adopt OT",
            "Kernel >> OT": "Ground metric noise dominates; stay with kernel",
            "OT ≈ Kernel": "Both capture same information; prefer kernel (simpler)",
        },
        "pending_steps": [
            "Compute full 894×894 Wasserstein distance matrix",
            "Aggregate O*NET distances to Census codes",
            "Run conditional logit comparison",
            "Report α coefficients and log-likelihoods",
        ],
    }

    # Save stub
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"wasserstein_comparison_v{VERSION.replace('.', '')}_stub.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Stub results saved to: {output_path}")


if __name__ == "__main__":
    main()
