#!/usr/bin/env python3
"""
Phase C: Feature Inspection

Validates whether SAE features are semantically coherent or
just efficient compression (PCA with extra steps).

Usage:
    PYTHONPATH=src python tests/run_phase_c.py
"""

from pathlib import Path
import sys
import json

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from task_space import build_dwa_activity_domain


def analyze_activation_distribution(features: np.ndarray) -> dict:
    """
    Analyze how activation is distributed across features.

    Power law = a few features doing all the work (dominant concepts)
    Uniform = noise or arbitrary clustering
    """
    # Total activation per feature (sum across all DWAs)
    feature_totals = features.sum(axis=0)

    # Filter to active features only
    active_mask = feature_totals > 0
    active_totals = feature_totals[active_mask]
    n_active = len(active_totals)

    # Sort descending
    sorted_totals = np.sort(active_totals)[::-1]

    # Compute concentration metrics
    total_activation = sorted_totals.sum()

    # Top-k concentration
    top_10_share = sorted_totals[:10].sum() / total_activation
    top_50_share = sorted_totals[:50].sum() / total_activation
    top_100_share = sorted_totals[:100].sum() / total_activation

    # Gini coefficient (0 = perfectly uniform, 1 = maximally concentrated)
    n = len(sorted_totals)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_totals) / (n * np.sum(sorted_totals))) - (n + 1) / n

    # Effective number of features (entropy-based)
    probs = sorted_totals / total_activation
    probs = probs[probs > 0]  # Remove zeros for log
    entropy = -np.sum(probs * np.log(probs))
    effective_n = np.exp(entropy)

    return {
        "n_active_features": int(n_active),
        "top_10_concentration": float(top_10_share),
        "top_50_concentration": float(top_50_share),
        "top_100_concentration": float(top_100_share),
        "gini_coefficient": float(gini),
        "effective_n_features": float(effective_n),
        "concentration_ratio": float(n_active / effective_n),  # >1 means power law
        "sorted_totals_top20": [float(x) for x in sorted_totals[:20]],
    }


def inspect_top_features(
    features: np.ndarray,
    domain,
    n_features: int = 20,
    n_top_dwas: int = 5,
) -> list[dict]:
    """
    Inspect the top N most active features.

    Returns list of feature reports with top DWAs for each.
    """
    # Total activation per feature
    feature_totals = features.sum(axis=0)

    # Get top features by total activation
    top_feature_indices = np.argsort(feature_totals)[::-1][:n_features]

    reports = []
    for rank, feat_idx in enumerate(top_feature_indices):
        # Get activations for this feature across all DWAs
        activations = features[:, feat_idx]

        # Get top DWAs for this feature
        top_dwa_indices = np.argsort(activations)[::-1][:n_top_dwas]

        top_dwas = []
        for dwa_idx in top_dwa_indices:
            dwa_id = domain.activity_ids[dwa_idx]
            dwa_title = domain.activity_names[dwa_id]
            activation = activations[dwa_idx]
            if activation > 0:  # Only include if actually activates
                top_dwas.append({
                    "dwa_id": dwa_id,
                    "title": dwa_title,
                    "activation": float(activation),
                })

        # Count how many DWAs activate this feature
        n_activating = (activations > 0.01).sum()

        reports.append({
            "rank": rank + 1,
            "feature_index": int(feat_idx),
            "total_activation": float(feature_totals[feat_idx]),
            "n_dwas_activating": int(n_activating),
            "top_dwas": top_dwas,
        })

    return reports


def assess_coherence(feature_reports: list[dict]) -> dict:
    """
    Assess semantic coherence of features.

    Heuristic checks:
    - Do top DWAs share obvious semantic themes?
    - Are features activating on reasonable numbers of DWAs?
    """
    assessments = []

    for report in feature_reports:
        top_dwas = report["top_dwas"]
        n_activating = report["n_dwas_activating"]

        # Extract just the titles for analysis
        titles = [d["title"] for d in top_dwas]

        # Simple heuristic: check for common words (excluding stopwords)
        stopwords = {"and", "or", "the", "a", "an", "to", "of", "for", "in", "on", "with", "by"}

        all_words = []
        for title in titles:
            words = title.lower().replace(",", "").replace(".", "").split()
            all_words.extend([w for w in words if w not in stopwords and len(w) > 2])

        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(3)

        # Coherence score: if top 3 words appear in multiple titles, likely coherent
        coherence_score = sum(count for _, count in most_common) / (len(titles) * 3) if titles else 0

        assessments.append({
            "feature_index": report["feature_index"],
            "n_activating": n_activating,
            "common_words": most_common,
            "coherence_score": coherence_score,
            "titles": titles,
        })

    # Overall assessment
    avg_coherence = np.mean([a["coherence_score"] for a in assessments])
    high_coherence_count = sum(1 for a in assessments if a["coherence_score"] > 0.3)

    return {
        "feature_assessments": assessments,
        "avg_coherence_score": float(avg_coherence),
        "high_coherence_features": high_coherence_count,
        "total_inspected": len(assessments),
    }


def main():
    print("=" * 60)
    print("Phase C: Feature Inspection")
    print("=" * 60)

    # Load data
    output_dir = Path("outputs/phase_c")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading sparse features and domain...")
    features = np.load("outputs/phase_b/dwa_sparse_features.npy")
    domain = build_dwa_activity_domain()
    print(f"      Features shape: {features.shape}")
    print(f"      Domain: {domain.n_activities} activities")

    # Activation distribution analysis
    print("\n[2/4] Analyzing feature activation distribution...")
    dist_analysis = analyze_activation_distribution(features)

    print(f"      Active features: {dist_analysis['n_active_features']}")
    print(f"      Effective N (entropy): {dist_analysis['effective_n_features']:.0f}")
    print(f"      Concentration ratio: {dist_analysis['concentration_ratio']:.2f}")
    print(f"      Gini coefficient: {dist_analysis['gini_coefficient']:.3f}")
    print(f"      Top 10 features hold: {dist_analysis['top_10_concentration']:.1%} of activation")
    print(f"      Top 50 features hold: {dist_analysis['top_50_concentration']:.1%} of activation")
    print(f"      Top 100 features hold: {dist_analysis['top_100_concentration']:.1%} of activation")

    # Interpretation
    if dist_analysis['gini_coefficient'] > 0.5:
        dist_interpretation = "POWER LAW - A few features dominate (suggests dominant concepts)"
    elif dist_analysis['gini_coefficient'] > 0.3:
        dist_interpretation = "MODERATE - Some concentration but reasonably spread"
    else:
        dist_interpretation = "UNIFORM - Activation spread evenly (suggests noise or over-fragmentation)"
    print(f"      Interpretation: {dist_interpretation}")

    # Feature inspection
    print("\n[3/4] Inspecting top 20 features...")
    feature_reports = inspect_top_features(features, domain, n_features=20, n_top_dwas=5)

    # Print human-readable report
    print("\n" + "-" * 60)
    print("TOP 20 FEATURES BY TOTAL ACTIVATION")
    print("-" * 60)

    for report in feature_reports:
        print(f"\nFeature #{report['rank']} (index {report['feature_index']})")
        print(f"  Total activation: {report['total_activation']:.2f}")
        print(f"  DWAs activating: {report['n_dwas_activating']}")
        print(f"  Top DWAs:")
        for dwa in report['top_dwas'][:5]:
            print(f"    - {dwa['title'][:60]}: {dwa['activation']:.3f}")

    # Coherence assessment
    print("\n[4/4] Assessing semantic coherence...")
    coherence = assess_coherence(feature_reports)

    print(f"\n      Average coherence score: {coherence['avg_coherence_score']:.3f}")
    print(f"      High-coherence features (>0.3): {coherence['high_coherence_features']} / {coherence['total_inspected']}")

    # Save results
    results = {
        "activation_distribution": dist_analysis,
        "feature_reports": feature_reports,
        "coherence_assessment": {
            "avg_coherence_score": coherence["avg_coherence_score"],
            "high_coherence_features": coherence["high_coherence_features"],
            "total_inspected": coherence["total_inspected"],
        },
        "interpretation": {
            "distribution": dist_interpretation,
            "pass_criteria": ">=10/20 features show semantic coherence",
        }
    }

    with open(output_dir / "feature_inspection.json", "w") as f:
        json.dump(results, f, indent=2)

    # Human-readable report
    with open(output_dir / "feature_interpretability.txt", "w") as f:
        f.write("PHASE C: FEATURE INTERPRETABILITY AUDIT\n")
        f.write("=" * 60 + "\n\n")

        f.write("ACTIVATION DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Active features: {dist_analysis['n_active_features']}\n")
        f.write(f"Effective N: {dist_analysis['effective_n_features']:.0f}\n")
        f.write(f"Gini coefficient: {dist_analysis['gini_coefficient']:.3f}\n")
        f.write(f"Top 10 hold: {dist_analysis['top_10_concentration']:.1%}\n")
        f.write(f"Top 50 hold: {dist_analysis['top_50_concentration']:.1%}\n")
        f.write(f"Interpretation: {dist_interpretation}\n\n")

        f.write("TOP 20 FEATURES\n")
        f.write("-" * 40 + "\n")
        for report in feature_reports:
            f.write(f"\nFeature #{report['rank']} (idx {report['feature_index']})\n")
            f.write(f"  Activation: {report['total_activation']:.2f}, DWAs: {report['n_dwas_activating']}\n")
            for dwa in report['top_dwas'][:5]:
                f.write(f"  - {dwa['title'][:55]}\n")

        f.write("\n\nCOHERENCE ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average coherence: {coherence['avg_coherence_score']:.3f}\n")
        f.write(f"High-coherence features: {coherence['high_coherence_features']}/{coherence['total_inspected']}\n")

    print(f"\n      Results saved to {output_dir}")

    # Decision
    print("\n" + "=" * 60)
    print("PHASE C VERDICT")
    print("=" * 60)

    passes = coherence['high_coherence_features'] >= 10

    if passes:
        print(f"✓ PASS: {coherence['high_coherence_features']}/20 features show semantic coherence")
        print("  → Features capture meaningful concepts")
        print("  → Proceed to Phase D (SAE Validation)")
    else:
        print(f"✗ FAIL: Only {coherence['high_coherence_features']}/20 features show coherence")
        print("  → SAE may be doing efficient compression, not meaningful decomposition")
        print("  → Consider: (a) retrain with different λ, (b) use supervised probing instead")

    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
