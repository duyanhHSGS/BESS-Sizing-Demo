from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "templates" / "index.html"


def _template() -> str:
    return TEMPLATE.read_text(encoding="utf-8")


def test_monthly_panel_is_at_bottom_of_benchmarking_tab() -> None:
    html = _template()
    detective = html.index('id="benchmarking-detective-area"')
    monthly = html.index('id="benchmarking-monthly"')
    live = html.index('id="tab-live-runs"')
    assert detective < monthly < live
    assert 'id="benchmarking-monthly-split-policy"' in html
    assert 'id="benchmarking-monthly-list"' in html


def test_monthly_ui_reuses_fixed_30_day_benchmark_blocks() -> None:
    html = _template()
    assert "const blocks = baseline?.summary?.blocks || [];" in html
    assert "const expectedEndDay = startDay + 29;" in html
    assert "days.length === 30 && actualEndDay === expectedEndDay" in html
    assert "Cut boundary after Day ${expectedEndDay}" in html


def test_partial_bucket_keeps_benchmark_coverage_separate_from_training_role() -> None:
    html = _template()
    assert "benchmark includes available days" in html
    assert "Partial buckets may still be benchmarked even when that checkpoint ignored them during training." in html
    assert "benchmarkIgnoredDay(meta, dayIndex)" in html
    assert 'return "ignored";' in html


def test_monthly_savings_are_derived_from_existing_no_bess_and_oracle_blocks() -> None:
    html = _template()
    assert 'blockMaps.get("no_bess")?.get(startDay)' in html
    assert 'blockMaps.get("oracle")?.get(startDay)' in html
    assert "noBessCost - cost" in html
    assert "cost - oracleCost" in html
    assert "noBessCost !== 0" in html
    assert "Saving vs No-BESS" in html
    assert "Gap to Oracle" in html


def test_split_roles_use_checkpoint_metadata_and_keep_legacy_unknown() -> None:
    html = _template()
    assert "meta.train_range" in html
    assert "meta.validation_range" in html
    assert "meta.test_range" in html
    assert "meta?.ignored_unselected_buckets" in html
    assert 'train: "🟢 TRAIN"' in html
    assert 'validation: "🟡 VALIDATION"' in html
    assert 'test: "🔵 TEST"' in html
    assert 'ignored: "✂️ IGNORED"' in html
    assert 'mixed: "🟣 MIXED / OVERLAP"' in html
    assert 'unknown: "⚪ ROLE UNKNOWN"' in html


def test_monthly_split_selector_is_fighter_specific() -> None:
    html = _template()
    assert "benchmarkMonthlyPolicyRows()" in html
    assert "contextByName.get(row.id)" in html
    assert "Split colors come only from ${selected.contestant.label} checkpoint metadata" in html
    assert 'benchmarkingDom.monthlySplitPolicy.addEventListener("change", renderBenchmarkMonthlyData);' in html


def test_completed_benchmark_result_renders_monthly_panel() -> None:
    html = _template()
    render_start = html.index("function renderBenchmarkResult()")
    render_end = html.index("function scopedBenchmarkRows()", render_start)
    render_body = html[render_start:render_end]
    assert "renderBenchmarkMonthlyData();" in render_body


def test_monthly_ui_has_architecture_todo_guard() -> None:
    html = _template()
    assert "TODO(BENCHMARK-MONTHLY-UI)" in html
    projarch = (ROOT / "git-plz-ignore" / "projarch.md").read_text(encoding="utf-8")
    assert "30-Day Monthly Data" in projarch
    assert "TODO(BENCHMARK-MONTHLY-UI)" in projarch
