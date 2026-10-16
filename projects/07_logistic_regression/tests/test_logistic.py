import numpy as np
import pytest


def test_quality_and_loss(project):
    result = project.experiment()
    assert result["final_loss"] < result["initial_loss"] * 0.4
    assert result["numpy_accuracy"] >= 0.90
    assert abs(result["numpy_accuracy"] - result["baseline_accuracy"]) <= 0.05


def test_prediction_shapes_and_probability_bounds(project):
    x, test_x, y, _ = project.dataset()
    model = project.LogisticModel(iterations=100).fit(x, y)
    assert model.predict(test_x).shape == (len(test_x),)
    assert model.weights.shape == (x.shape[1],)
    assert np.all(
        (model.predict_proba(test_x) >= 0) & (model.predict_proba(test_x) <= 1)
    )
    assert np.isfinite(project.sigmoid(np.array([-1000.0, 1000.0]))).all()


def test_gradient_finite_difference(project):
    x = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 1.0]])
    y, weights = np.array([1.0, 0.0, 1.0]), np.array([0.2, -0.1])
    _, gradient, bias_gradient = project.objective_gradient(x, y, weights, 0.1)
    h = 1e-6
    for index in range(2):
        step = np.zeros(2)
        step[index] = h
        numeric = (
            project.objective_gradient(x, y, weights + step, 0.1)[0]
            - project.objective_gradient(x, y, weights - step, 0.1)[0]
        ) / (2 * h)
        assert gradient[index] == pytest.approx(numeric, abs=1e-7)
    numeric = (
        project.objective_gradient(x, y, weights, 0.1 + h)[0]
        - project.objective_gradient(x, y, weights, 0.1 - h)[0]
    ) / (2 * h)
    assert bias_gradient == pytest.approx(numeric, abs=1e-7)


def test_invalid_and_unfitted(project):
    with pytest.raises(ValueError, match="Fit"):
        project.LogisticModel().predict(np.ones((1, 2)))
    for x, y in [(np.empty((0, 2)), np.array([])), (np.ones((2, 2)), np.array([0, 2]))]:
        with pytest.raises(ValueError):
            project.LogisticModel().fit(x, y)


def test_repeated_fit_resets_parameters(project):
    x, _, y, _ = project.dataset()
    model = project.LogisticModel(iterations=30).fit(x, y)
    weights = model.weights.copy()
    model.fit(x, y)
    np.testing.assert_allclose(model.weights, weights)


def test_score_validates_labels(project):
    x, test_x, y, test_y = project.dataset()
    model = project.LogisticModel(iterations=10).fit(x, y)
    assert 0 <= model.score(test_x, test_y) <= 1
    with pytest.raises(ValueError):
        model.score(test_x, np.zeros((len(test_y), 1)))
    with pytest.raises(ValueError):
        model.score(test_x, np.full(len(test_y), 2))
