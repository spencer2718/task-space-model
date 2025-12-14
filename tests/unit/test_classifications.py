from pathlib import Path
from collections import Counter
import pytest


class TestGWAClassification:
    """Tests for GWA classification (Task 3.1)."""

    def test_gwa_classification_coverage(self):
        """All 41 GWAs should be classified."""
        from task_space.data.classifications import get_gwa_classifications

        gwa_classes = get_gwa_classifications(Path("data/onet/db_30_0_excel"))

        assert len(gwa_classes) == 41, f"Expected 41 GWAs, got {len(gwa_classes)}"

        # Check distribution
        counts = Counter(gwa_classes.values())
        print(f"Classification distribution: {counts}")

        assert counts['cognitive'] >= 10, "Should have ≥10 cognitive GWAs"
        assert counts['physical'] >= 3, "Should have ≥3 physical GWAs"
        assert counts['interpersonal'] >= 10, "Should have ≥10 interpersonal GWAs"

    def test_gwa_classification_parsing(self):
        """Classification should use dot-parsing, not fixed slicing."""
        from task_space.data.classifications import classify_gwa

        # Standard cases
        assert classify_gwa('4.A.1.a.1') == 'cognitive'
        assert classify_gwa('4.A.2.b.2') == 'cognitive'
        assert classify_gwa('4.A.3.a.1') == 'physical'
        assert classify_gwa('4.A.3.b.4') == 'technical'
        assert classify_gwa('4.A.4.a.1') == 'interpersonal'

        # Edge case: hypothetical double-digit segment (robustness check)
        # If O*NET ever adds 4.A.10.*, this should not break
        with pytest.raises(ValueError):
            classify_gwa('4.A.10.a.1')  # Currently invalid, but parsing should handle gracefully

    def test_gwa_invalid_id_raises(self):
        """Invalid IDs should raise ValueError."""
        from task_space.data.classifications import classify_gwa

        with pytest.raises(ValueError):
            classify_gwa('invalid')

        with pytest.raises(ValueError):
            classify_gwa('4.B.1.a.1')  # Not 4.A.*

        with pytest.raises(ValueError):
            classify_gwa('4.A')  # Too short


class TestDWAClassification:
    """Tests for DWA classification propagation (Task 3.2)."""

    def test_dwa_classification_propagation(self):
        """All 2087 DWAs should inherit parent GWA classification."""
        from task_space.data.classifications import get_dwa_classifications

        dwa_classes = get_dwa_classifications(Path("data/onet/db_30_0_excel"))

        assert len(dwa_classes) >= 2000, f"Expected ~2087 DWAs, got {len(dwa_classes)}"

        # Verify hierarchy consistency via dot-parsing
        for dwa_id, category in list(dwa_classes.items())[:20]:
            parts = dwa_id.split('.')
            segment = parts[2]  # GWA category segment

            if segment in ('1', '2'):
                assert category == 'cognitive', f"{dwa_id} should be cognitive"
            elif segment == '3':
                sub = parts[3]
                expected = 'physical' if sub == 'a' else 'technical'
                assert category == expected, f"{dwa_id} should be {expected}"
            elif segment == '4':
                assert category == 'interpersonal', f"{dwa_id} should be interpersonal"

    def test_dwa_parent_extraction(self):
        """Parent GWA extraction should use dot-parsing."""
        from task_space.data.classifications import _extract_parent_gwa

        assert _extract_parent_gwa('4.A.1.a.1.I01.D01') == '4.A.1.a.1'
        assert _extract_parent_gwa('4.A.3.b.4.I02.D03') == '4.A.3.b.4'

        with pytest.raises(ValueError):
            _extract_parent_gwa('4.A.1.a')  # Too short


class TestRoutineScores:
    """Tests for routine score functions (Task 3.3)."""

    def test_routine_scores_sensible(self):
        """Routine scores should have reasonable distribution."""
        from task_space.data.classifications import get_routine_scores
        import numpy as np

        scores = get_routine_scores(Path("data/onet/db_30_0_excel"))
        values = np.array(list(scores.values()))

        assert len(scores) >= 700, f"Expected ≥700 occupations, got {len(scores)}"
        # O*NET Work Context uses importance scale (typically 1-5 range, aggregated)
        # Actual range is approximately 3-18 based on Data Value field
        assert 1 < values.mean() < 50, f"Mean {values.mean():.1f} seems off"
        assert values.std() > 0.5, f"Std {values.std():.1f} too low (no variation)"
        assert values.max() > values.min(), "Should have variation in scores"

    def test_projected_routine_labeled_correctly(self):
        """Projected routine function name should indicate endogeneity."""
        from task_space.data import classifications

        # Function should be named to indicate projection, not intrinsic
        assert hasattr(classifications, 'get_activity_projected_routine_scores')
        assert not hasattr(classifications, 'get_activity_routine_scores')  # Should NOT exist

        # Docstring should contain warning
        docstring = classifications.get_activity_projected_routine_scores.__doc__
        assert 'ENDOGENEITY' in docstring.upper() or 'PROJECTED' in docstring.upper()

    def test_projected_routine_computation(self):
        """Projected routine should compute weighted average."""
        from task_space.data.classifications import get_activity_projected_routine_scores
        import numpy as np

        onet_path = Path("data/onet/db_30_0_excel")

        # Simple test case
        activity_ids = ['test_a', 'test_b']
        occupation_matrix = np.array([
            [1.0, 0.0],  # Occ 0 only uses activity 0
            [0.0, 1.0],  # Occ 1 only uses activity 1
        ])
        occupation_codes = ['11-1011.00', '11-1021.00']

        # This will use the real routine scores from O*NET
        # Just verify it returns the right shape and reasonable values
        result = get_activity_projected_routine_scores(
            onet_path, activity_ids, occupation_matrix, occupation_codes
        )

        assert result.shape == (2,)
        assert np.all(result >= 0), "Routine scores should be non-negative"
