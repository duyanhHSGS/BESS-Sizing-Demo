from bess.paths import PROJECT_ROOT


def test_dashboard_has_exactly_the_seven_surviving_tabs() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    for tab in ("sizing", "training", "human", "benchmark", "live", "shadow", "dispatch"):
        assert f'id="tab-{tab}"' in html
    assert "/api/brain1/" not in html
    assert "/api/brain2/" not in html
    assert "weather" not in html.lower()


def test_dispatch_exposes_builtin_brains_and_checkpoint_roster() -> None:
    source = (PROJECT_ROOT / "bess" / "webapp.py").read_text(encoding="utf-8")
    assert '"id": "brain1"' in source
    assert '"id": "brain2"' in source
    assert "*compatible" in source
    assert "list_compatible_checkpoints" in source


def test_dense_viewers_keep_toggleable_lines_and_peak_overlays() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    javascript = (PROJECT_ROOT / "web" / "brain_hq.js").read_text(encoding="utf-8")
    for identifier in (
        "dispatch-power-lines",
        "dispatch-action-lines",
        "dispatch-soc-lines",
        "dispatch-money-lines",
        "live-lines",
        "shadow-lines",
        "human-lines",
        "sizing-lines",
    ):
        assert f'id="{identifier}"' in html
    assert "controlledPlot" in javascript
    assert "bindChartHover" in javascript
    assert "Raw billing-scope peak" in javascript
    assert "requested_battery_kw" in javascript
    assert "projected_battery_kw" in javascript
    assert "executed_battery_kw" in javascript
    assert "Oracle grid" in javascript
    assert "Oracle billing peak" in javascript
    assert "Oracle battery schedule" in javascript
    assert "Oracle SOC" in javascript
    assert '<pre id="human-status"' not in html
    assert '<pre id="live-status"' not in html
    assert '<pre id="dispatch-status"' not in html
