import pytest


def test_hand_calculated_waits(project):
    visits = project.simulate(
        [project.Customer(0, 3), project.Customer(1, 2), project.Customer(5, 1)]
    )
    assert [v.wait for v in visits] == [0, 2, 0]
    assert project.summarize(visits).mean_wait == pytest.approx(2 / 3)
    assert project.summarize(visits).utilization == 1


def test_same_time_arrivals_and_zero_service(project):
    visits = project.simulate(
        [project.Customer(0, 0), project.Customer(0, 2), project.Customer(0, 1)]
    )
    assert [v.start for v in visits] == [0, 0, 2]


def test_empty_and_invalid(project):
    assert project.summarize(project.simulate([])).count == 0
    with pytest.raises(ValueError):
        project.simulate([project.Customer(-1, 1)])
    with pytest.raises(ValueError):
        project.generate_customers(1, 0, 1)


def test_seeded_reproducibility_and_idle_time(project):
    first = project.generate_customers(50, 0.5, 1, 3)
    assert first == project.generate_customers(50, 0.5, 1, 3)
    visits = project.simulate([project.Customer(0, 1), project.Customer(10, 1)])
    assert project.summarize(visits).utilization == pytest.approx(2 / 11)


def test_unsorted_input_keeps_customer_identity(project):
    visits = project.simulate([project.Customer(4, 1), project.Customer(0, 6)])
    assert [v.start for v in visits] == [6, 0]
