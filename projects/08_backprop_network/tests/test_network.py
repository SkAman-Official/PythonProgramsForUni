import numpy as np
import pytest


def test_loss_and_generalization(project):
    result = project.experiment()
    assert result["final_loss"] < result["initial_loss"] * 0.4
    assert result["test_accuracy"] >= 0.85


def test_shapes(project):
    x, test_x, y, _ = project.dataset()
    model = project.Network(hidden_units=5, iterations=5).fit(x, y)
    assert model.parameters["w1"].shape == (2, 5)
    assert model.parameters["w2"].shape == (5, 1)
    assert model.predict_proba(test_x).shape == (len(test_x),)
    assert model.predict(test_x).shape == (len(test_x),)


def test_all_backprop_parameters(project):
    x = np.array([[0.2, -0.1], [0.5, 0.7], [-0.3, 0.2]])
    y = np.array([0.0, 1.0, 0.0])
    parameters = project.Network(hidden_units=3).initialize(2)
    _, gradients = project.loss_and_gradients(x, y, parameters)
    h = 1e-6
    for name, values in parameters.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + h
            plus = project.loss_and_gradients(x, y, parameters)[0]
            values[index] = original - h
            minus = project.loss_and_gradients(x, y, parameters)[0]
            values[index] = original
            assert gradients[name][index] == pytest.approx(
                (plus - minus) / (2 * h), abs=1e-7
            )


def test_invalid_input_and_unfitted(project):
    with pytest.raises(ValueError):
        project.Network().predict(np.ones((1, 2)))
    with pytest.raises(ValueError):
        project.Network().fit(np.array([[np.nan, 1]]), np.array([0]))
    with pytest.raises(ValueError):
        project.Network(hidden_units=0).initialize(2)


def test_deterministic_training(project):
    x, _, y, _ = project.dataset()
    first = project.Network(iterations=10).fit(x, y)
    second = project.Network(iterations=10).fit(x, y)
    np.testing.assert_array_equal(first.parameters["w1"], second.parameters["w1"])


def test_score_validates_labels(project):
    x, test_x, y, test_y = project.dataset()
    model = project.Network(iterations=10).fit(x, y)
    assert 0 <= model.score(test_x, test_y) <= 1
    with pytest.raises(ValueError):
        model.score(test_x, np.zeros((len(test_y), 1)))
    with pytest.raises(ValueError):
        model.score(test_x, np.full(len(test_y), 2))
