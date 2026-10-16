import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score


def test_shapes_loss_and_recovery(project):
    x, truth = project.dataset()
    result = project.kmeans(x)
    assert result.centers.shape == (3, 5)
    assert result.labels.shape == (len(x),)
    assert result.losses[-1] < result.losses[0]
    assert np.all(np.diff(result.losses) <= 1e-8)
    assert adjusted_rand_score(truth, result.labels) >= 0.90


def test_pca_geometry(project):
    x, _ = project.dataset()
    projected, explained = project.project_pca(x)
    assert projected.shape == (len(x), 2)
    assert explained.shape == (2,)
    np.testing.assert_allclose(projected.mean(axis=0), 0, atol=1e-12)
    assert 0 < explained.sum() <= 1


def test_duplicate_points_empty_clusters(project):
    result = project.kmeans(np.ones((5, 2)), count=3)
    assert result.losses[-1] == 0
    assert np.isfinite(result.centers).all()
    projected, explained = project.project_pca(np.ones((5, 2)))
    assert not projected.any() and not explained.any()


def test_invalid_inputs(project):
    for x, count in [
        (np.empty((0, 2)), 1),
        (np.ones((2, 2)), 3),
        (np.array([[np.nan]]), 1),
    ]:
        with pytest.raises(ValueError):
            project.kmeans(x, count)
    with pytest.raises(ValueError):
        project.project_pca(np.ones((3, 2)), 3)


def test_plot_output_and_repeatability(project, tmp_path):
    x, _ = project.dataset()
    result = project.kmeans(x)
    np.testing.assert_array_equal(result.labels, project.kmeans(x).labels)
    output = tmp_path / "clusters.png"
    project.save_plot(x, result, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000
