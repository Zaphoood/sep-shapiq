# ruff: noqa
"""Test to compare all TKNN implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import RadiusNeighborsClassifier

# TODO @murscht: get rid of print statements
# TODO @murscht: ignore linter errors for reference implementation


def test_tknn_implementations_comparison():
    """Test that all TKNN implementations return similar Shapley values."""
    # Create synthetic test data
    np.random.seed(42)
    X_train = np.random.randn(20, 3)
    y_train = np.random.randint(0, 3, size=20)
    x_test = np.array([0.5, -0.2, 0.1])

    # Create RadiusNeighborsClassifier
    model = RadiusNeighborsClassifier(radius=1.5)
    model.fit(X_train, y_train)

    # Initialize all implementations
    from shapiq_student.explainer.knn import TKNNExplainer, TKNNExplainerEfficient

    # Reference implementation from https://github.com/Jiachen-T-Wang/TKNN-Shapley/blob/main/helper_knn.py
    # Change default dis_metric from cosine to euclidean
    # Change distance comparison from < to <= as per paper definition
    def tnn_shapley_single(
        x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric="euclidean"
    ):
        N = len(y_train_few)
        sv = np.zeros(N)

        C = max(y_train_few) + 1
        if dis_metric == "cosine":
            distance = -np.dot(x_train_few, x_test) / np.linalg.norm(x_train_few, axis=1)
        else:
            distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
        Itau = (distance <= tau).nonzero()[0]

        Ct = len(Itau) + 1  # I implemented this adding +1
        Ca = np.sum(y_train_few[Itau] == y_test)

        reusable_sum = 0
        stable_ratio = 1
        for j in range(N):
            stable_ratio *= (N - j - Ct) / (N - j)
            reusable_sum += (1 / (j + 1)) * (1 - stable_ratio)
            # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

        for i in Itau:
            sv[i] = (int(y_test == y_train_few[i]) - 1 / C) / Ct
            if Ct >= 2:
                ca = Ca - int(y_test == y_train_few[i])
                sv[i] += (int(y_test == y_train_few[i]) / Ct - ca / (Ct * (Ct - 1))) * (
                    reusable_sum - 1
                )

        return sv

    # Get predictions and compute Shapley values
    y_test_pred = model.predict(x_test.reshape(1, -1))[0]
    class_index = np.where(model.classes_ == y_test_pred)[0][0]

    explainer_a1a2 = TKNNExplainer(model, class_index=class_index)
    explainer_efficient = TKNNExplainerEfficient(model, class_index=class_index)
    sv_reference = tnn_shapley_single(X_train, y_train, x_test, y_test_pred, tau=1.5)
    sv_a1a2 = explainer_a1a2.explain(x_test).values
    sv_efficient = explainer_efficient.explain(x_test).values

    # Comparison function with tolerance
    def compare_implementations(
        sv1: np.ndarray, sv2: np.ndarray, sv3: np.ndarray, atol: float = 1e-6, rtol: float = 1e-5
    ) -> dict[str, Any]:
        """Compare three Shapley value implementations."""
        results = {
            "a1a2_vs_efficient": {
                "close": np.allclose(sv1, sv2, atol=atol, rtol=rtol),
                "max_diff": np.max(np.abs(sv1 - sv2)),
                "mean_diff": np.mean(np.abs(sv1 - sv2)),
            },
            "a1a2_vs_reference": {
                "close": np.allclose(sv1, sv3, atol=atol, rtol=rtol),
                "max_diff": np.max(np.abs(sv1 - sv3)),
                "mean_diff": np.mean(np.abs(sv1 - sv3)),
            },
            "efficient_vs_reference": {
                "close": np.allclose(sv2, sv3, atol=atol, rtol=rtol),
                "max_diff": np.max(np.abs(sv2 - sv3)),
                "mean_diff": np.mean(np.abs(sv2 - sv3)),
            },
            "shapley_values": {"a1a2": sv1, "efficient": sv2, "reference": sv3},
            "sum_check": {
                "a1a2_sum": np.sum(sv1),
                "efficient_sum": np.sum(sv2),
                "reference_sum": np.sum(sv3),
            },
        }

        return results

    # Run comparison with different tolerance levels
    tolerances = [1e-10, 1e-8, 1e-6, 1e-4]

    for tol in tolerances:
        results = compare_implementations(sv_a1a2, sv_efficient, sv_reference, atol=tol)

        print(f"\n=== Tolerance: {tol} ===")
        print(
            f"A1/A2 vs Efficient: {results['a1a2_vs_efficient']['close']} "
            f"(max_diff: {results['a1a2_vs_efficient']['max_diff']:.2e})"
        )
        print(
            f"A1/A2 vs Reference: {results['a1a2_vs_reference']['close']} "
            f"(max_diff: {results['a1a2_vs_reference']['max_diff']:.2e})"
        )
        print(
            f"Efficient vs Reference: {results['efficient_vs_reference']['close']} "
            f"(max_diff: {results['efficient_vs_reference']['max_diff']:.2e})"
        )

        # If any comparison passes at this tolerance, break
        if (
            results["a1a2_vs_efficient"]["close"]
            and results["a1a2_vs_reference"]["close"]
            and results["efficient_vs_reference"]["close"]
        ):
            print(f"✓ All implementations agree within tolerance {tol}")
            break
    else:
        print("Implementations differ beyond tested tolerances")

        # Print detailed comparison
        print("\nShapley Values:")
        print(f"A1/A2:      {sv_a1a2}")
        print(f"Efficient:  {sv_efficient}")
        print(f"Reference:  {sv_reference}")

        print("\nSums:")
        print(f"A1/A2 sum:      {np.sum(sv_a1a2):.6f}")
        print(f"Efficient sum:  {np.sum(sv_efficient):.6f}")
        print(f"Reference sum:  {np.sum(sv_reference):.6f}")

        # Compare implementations with tolerance
        atol = 1e-6
        rtol = 1e-5
        assert np.allclose(sv_a1a2, sv_efficient, atol=atol, rtol=rtol), (
            "A1/A2 and Efficient implementations differ"
        )
        assert np.allclose(sv_a1a2, sv_reference, atol=atol, rtol=rtol), (
            "A1/A2 and Reference implementations differ"
        )
        assert np.allclose(sv_efficient, sv_reference, atol=atol, rtol=rtol), (
            "Efficient and Reference implementations differ"
        )


if __name__ == "__main__":
    test_tknn_implementations_comparison()
