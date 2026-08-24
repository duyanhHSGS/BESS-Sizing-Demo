from __future__ import annotations

from webui import create_app


def test_original_ui_renders_and_compat_routes_boot() -> None:
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    index = client.get("/")
    assert index.status_code == 200
    assert b"<title>Sizing Demo</title>" in index.data
    for label in (
        b"Sizing Demo",
        b"Get Weather",
        b"Training Lab",
        b"Benchmarking",
        b"Live Runs",
        b"Shadow Running",
        b"Dispatch Viewer",
    ):
        assert label in index.data

    datasets = client.get("/api/training/datasets")
    assert datasets.status_code == 200
    assert isinstance(datasets.get_json(), list)

    checkpoints = client.get("/api/training/checkpoints")
    assert checkpoints.status_code == 200
    assert isinstance(checkpoints.get_json(), list)

    dispatch = client.get("/api/dispatch/policies")
    assert dispatch.status_code == 200
    assert "policies" in dispatch.get_json()

    benchmark = client.get("/api/benchmarking/context")
    assert benchmark.status_code == 200
    assert "oracle_ready" in benchmark.get_json()

    live = client.get("/api/live-runs")
    assert live.status_code == 200
    assert "sessions" in live.get_json()

    shadow = client.get("/api/shadow/config")
    assert shadow.status_code == 200
    assert "source_kind" in shadow.get_json()

    weather = client.get("/api/weather/context")
    assert weather.status_code == 200
    assert "datasets" in weather.get_json()
