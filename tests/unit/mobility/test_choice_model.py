"""
Unit tests for conditional logit choice model.

Tests:
- Coefficient recovery from synthetic data
- Choice dataset construction
- Result dataclass functionality
- Assumptions tracking
"""

import numpy as np
import pandas as pd
import pytest

from task_space.mobility.choice_model import (
    ChoiceModelResult,
    build_choice_dataset,
    fit_conditional_logit,
    compute_odds_ratios,
)


class TestChoiceDatasetConstruction:
    """Test choice dataset building."""

    def test_choice_set_size(self):
        """Each transition should have correct number of alternatives."""
        # Create synthetic transitions
        n_transitions = 100
        transitions_df = pd.DataFrame({
            "origin_occ": np.random.choice([1, 2, 3], n_transitions),
            "dest_occ": np.random.choice([4, 5, 6], n_transitions),
        })

        # Create distance matrices
        occ_codes = [1, 2, 3, 4, 5, 6]
        n_occ = len(occ_codes)
        d_sem = np.random.rand(n_occ, n_occ)
        d_inst = np.random.rand(n_occ, n_occ)

        n_alts = 3
        choice_df = build_choice_dataset(
            transitions_df, d_sem, d_inst, occ_codes,
            n_alternatives=n_alts,
        )

        # Should have (n_alts + 1) rows per transition
        n_groups = choice_df["transition_id"].nunique()
        expected_rows = n_groups * (n_alts + 1)
        assert len(choice_df) == expected_rows

    def test_chosen_flag_correct(self):
        """Each choice set should have exactly one chosen option."""
        transitions_df = pd.DataFrame({
            "origin_occ": [1, 2],
            "dest_occ": [3, 4],
        })

        occ_codes = [1, 2, 3, 4, 5]
        n_occ = len(occ_codes)
        d_sem = np.random.rand(n_occ, n_occ)
        d_inst = np.random.rand(n_occ, n_occ)

        choice_df = build_choice_dataset(
            transitions_df, d_sem, d_inst, occ_codes,
            n_alternatives=2,
        )

        # Check each group has exactly one chosen
        chosen_counts = choice_df.groupby("transition_id")["chosen"].sum()
        assert all(chosen_counts == 1), "Each choice set should have exactly one chosen"

    def test_distances_negated(self):
        """Covariates should be negated distances."""
        transitions_df = pd.DataFrame({
            "origin_occ": [1],
            "dest_occ": [2],
        })

        occ_codes = [1, 2, 3]
        # Known distance values
        d_sem = np.array([
            [0.0, 0.5, 0.8],
            [0.5, 0.0, 0.3],
            [0.8, 0.3, 0.0],
        ])
        d_inst = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ])

        choice_df = build_choice_dataset(
            transitions_df, d_sem, d_inst, occ_codes,
            n_alternatives=1,
            random_seed=42,
        )

        # Check chosen destination (occ=2, from origin=1)
        chosen_row = choice_df[choice_df["chosen"] == 1].iloc[0]
        assert chosen_row["neg_d_sem"] == -0.5  # d_sem[0,1] = 0.5
        assert chosen_row["neg_d_inst"] == -1.0  # d_inst[0,1] = 1.0


class TestConditionalLogitFit:
    """Test model fitting."""

    def test_recovers_known_coefficients(self):
        """Model should approximately recover known generating coefficients."""
        np.random.seed(42)

        # True coefficients (for negated distances)
        true_alpha = 2.0
        true_beta = 0.5

        # Generate synthetic choice data
        n_transitions = 500
        n_alts = 5
        n_occ = 20

        rows = []
        for tid in range(n_transitions):
            origin_idx = np.random.randint(0, n_occ)

            # Generate distances
            d_sems = np.random.rand(n_alts + 1) * 2
            d_insts = np.random.rand(n_alts + 1) * 3

            # Compute utilities
            utilities = true_alpha * (-d_sems) + true_beta * (-d_insts)
            probs = np.exp(utilities - utilities.max())  # Softmax
            probs = probs / probs.sum()

            # Sample chosen
            chosen_idx = np.random.choice(n_alts + 1, p=probs)

            for j in range(n_alts + 1):
                rows.append({
                    "transition_id": tid,
                    "occ": j,
                    "chosen": 1 if j == chosen_idx else 0,
                    "neg_d_sem": -d_sems[j],
                    "neg_d_inst": -d_insts[j],
                })

        choice_df = pd.DataFrame(rows)
        result = fit_conditional_logit(choice_df)

        # Should recover coefficients within tolerance
        # (larger samples would give tighter bounds)
        assert abs(result.alpha - true_alpha) < 0.5, \
            f"Alpha {result.alpha} not close to true {true_alpha}"
        assert abs(result.beta - true_beta) < 0.3, \
            f"Beta {result.beta} not close to true {true_beta}"

    def test_positive_coefficients_for_distance_preference(self):
        """Positive coefficients mean workers prefer lower distances."""
        np.random.seed(123)

        # Generate data where workers strongly prefer lower distances
        n_transitions = 200
        rows = []

        for tid in range(n_transitions):
            # Always choose the alternative with lowest distance
            d_sems = np.random.rand(3) * 2
            d_insts = np.random.rand(3) * 3
            total_dist = d_sems + d_insts
            chosen_idx = np.argmin(total_dist)

            for j in range(3):
                rows.append({
                    "transition_id": tid,
                    "occ": j,
                    "chosen": 1 if j == chosen_idx else 0,
                    "neg_d_sem": -d_sems[j],
                    "neg_d_inst": -d_insts[j],
                })

        choice_df = pd.DataFrame(rows)
        result = fit_conditional_logit(choice_df)

        # Both coefficients should be positive
        assert result.alpha > 0, "Alpha should be positive (prefer lower semantic distance)"
        assert result.beta > 0, "Beta should be positive (prefer lower institutional distance)"


class TestChoiceModelResult:
    """Test result dataclass."""

    def test_assumptions_populated(self):
        """Assumptions list should be populated by default."""
        result = ChoiceModelResult(
            alpha=1.0, alpha_se=0.1, alpha_t=10.0, alpha_p=0.0,
            beta=0.5, beta_se=0.05, beta_t=10.0, beta_p=0.0,
            log_likelihood=-1000.0,
            n_transitions=100, n_choice_rows=1100, n_alternatives=11,
            converged=True,
        )

        assert len(result.assumptions) > 0
        assert "IIA" in result.assumptions[0]

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all fields."""
        result = ChoiceModelResult(
            alpha=1.0, alpha_se=0.1, alpha_t=10.0, alpha_p=0.0,
            beta=0.5, beta_se=0.05, beta_t=10.0, beta_p=0.0,
            log_likelihood=-1000.0,
            n_transitions=100, n_choice_rows=1100, n_alternatives=11,
            converged=True,
        )

        d = result.to_dict()
        assert "alpha_coef" in d
        assert "beta_coef" in d
        assert "n_transitions" in d
        assert "assumptions" in d

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve values."""
        result = ChoiceModelResult(
            alpha=2.994, alpha_se=0.030, alpha_t=98.53, alpha_p=0.0,
            beta=0.215, beta_se=0.003, beta_t=63.42, beta_p=0.0,
            log_likelihood=-205528.9,
            n_transitions=89329, n_choice_rows=982619, n_alternatives=11,
            converged=True,
        )

        path = tmp_path / "test_results.json"
        result.save(str(path))
        loaded = ChoiceModelResult.load(str(path))

        assert abs(loaded.alpha - result.alpha) < 1e-6
        assert abs(loaded.beta - result.beta) < 1e-6
        assert loaded.n_transitions == result.n_transitions


class TestComputeOddsRatios:
    """Test odds ratio computation."""

    def test_odds_ratio_interpretation(self):
        """Odds ratios should be < 1 for positive coefficients."""
        result = ChoiceModelResult(
            alpha=2.0, alpha_se=0.1, alpha_t=20.0, alpha_p=0.0,
            beta=0.5, beta_se=0.05, beta_t=10.0, beta_p=0.0,
            log_likelihood=-1000.0,
            n_transitions=100, n_choice_rows=1100, n_alternatives=11,
            converged=True,
        )

        odds = compute_odds_ratios(result)

        # exp(-alpha) and exp(-beta) should be < 1
        assert odds["semantic"]["odds_ratio"] < 1
        assert odds["institutional"]["odds_ratio"] < 1

    def test_coefficient_ratio(self):
        """Coefficient ratio should be alpha/beta."""
        result = ChoiceModelResult(
            alpha=3.0, alpha_se=0.1, alpha_t=30.0, alpha_p=0.0,
            beta=1.5, beta_se=0.05, beta_t=30.0, beta_p=0.0,
            log_likelihood=-1000.0,
            n_transitions=100, n_choice_rows=1100, n_alternatives=11,
            converged=True,
        )

        odds = compute_odds_ratios(result)
        assert abs(odds["coefficient_ratio"] - 2.0) < 1e-6
