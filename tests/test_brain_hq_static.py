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
    assert "DEADLY monthly peak" in javascript
    assert "requested_battery_kw" in javascript
    assert "projected_battery_kw" in javascript
    assert "executed_battery_kw" in javascript
    assert "Oracle grid" in javascript
    assert "Post-Oracle DEADLY peak" in javascript
    assert "Oracle battery schedule" in javascript
    assert "Oracle SOC" in javascript
    assert "periodGroups" in javascript
    assert "row.billing_period" in javascript
    assert "Monthly factory + Oracle ceiling" in html
    assert "whole billing-period episodes" in html
    assert '<pre id="human-status"' not in html
    assert '<pre id="live-status"' not in html
    assert '<pre id="dispatch-status"' not in html


def test_og_visual_language_is_the_only_served_style_layer() -> None:
    html = (PROJECT_ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "web" / "brain_hq_overrides.css").read_text(encoding="utf-8")
    assert "/assets/brain_hq_overrides.css" in html
    assert "/assets/brain_hq.css" not in html
    assert "adapted from web/old.html" in css
    assert ".bill-strip" in css
    assert "#0f1419" in css


def test_oracle_and_brains_share_whole_billing_period_membership() -> None:
    oracle = (PROJECT_ROOT / "bess" / "evaluation" / "oracle" / "oracle_lp.py").read_text(encoding="utf-8")
    cache = (PROJECT_ROOT / "bess" / "evaluation" / "oracle" / "oracle_cache.py").read_text(encoding="utf-8")
    assert "_canonical_billing_days" in oracle
    assert 'day["billing_period"] = billing_period' in oracle
    assert "CACHE_VERSION = 4" in cache
