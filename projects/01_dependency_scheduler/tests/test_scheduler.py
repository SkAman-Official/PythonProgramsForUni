import json

import pytest


def test_parallel_schedule(project):
    slots = {slot.name: slot for slot in project.schedule(project.demo_jobs())}
    assert slots["analyse"].start == slots["review"].start == 2
    assert slots["report"].finish == 7
    assert project.critical_path(project.demo_jobs()) == [
        "prepare",
        "analyse",
        "report",
    ]


def test_cycle_and_missing_dependency(project):
    with pytest.raises(ValueError, match="cycle"):
        project.schedule([project.Job("a", 1, ("b",)), project.Job("b", 1, ("a",))])
    with pytest.raises(ValueError, match="Unknown"):
        project.schedule([project.Job("a", 1, ("missing",))])


def test_empty_and_zero_duration(project):
    assert project.schedule([]) == []
    assert project.critical_path([]) == []
    assert project.schedule([project.Job("zero", 0)])[0].finish == 0


def test_invalid_jobs(project):
    for jobs in (
        [project.Job("a", -1)],
        [project.Job("a", float("nan"))],
        [project.Job("a", 1), project.Job("a", 2)],
    ):
        with pytest.raises(ValueError):
            project.schedule(jobs)


def test_json_loading(project, tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps([{"name": "a", "duration": 2}]))
    assert project.load_jobs(path) == [project.Job("a", 2)]
    path.write_text('[{"name": "a", "duration": true}]')
    with pytest.raises(ValueError):
        project.load_jobs(path)
