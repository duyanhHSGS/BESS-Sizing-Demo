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
