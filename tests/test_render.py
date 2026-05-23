"""Tests — see module name for scope."""

from __future__ import annotations

from pathlib import Path

from biopsy.demo import write_demo_csv
from biopsy.profile import profile
from biopsy.render.html import render as render_html


def test_html_findings_groups_by_severity(tmp_path: Path) -> None:
    """Findings section carries severity/category data attributes so the
    in-page filter chips work without JS state."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    out = tmp_path / "report.html"
    render_html(prof, out)
    text = out.read_text()
    assert 'id="findings-filter"' in text
    # At least one finding per severity carries the data-sev attribute.
    assert 'data-sev="critical"' in text or 'data-sev="warning"' in text
    assert 'data-cat="' in text
    # Filter chips for severities exist (depend on what fired).
    assert 'class="chip' in text


def test_html_report_has_feature_drilldown(tmp_path: Path) -> None:
    """The HTML report renders a `<details class="feature-card panel">`
    per shortlisted feature."""
    csv = write_demo_csv(tmp_path / "demo.csv", n=1500)
    prof = profile(csv, target="churned")
    out = tmp_path / "report.html"
    render_html(prof, out)
    text = out.read_text()
    assert "feature-card" in text
    # At least one shortlisted feature appears inside a card.
    assert prof.clusters is not None and prof.clusters.shortlist
    feat = prof.clusters.shortlist[0].feature
    assert feat in text


def test_compare_html_renders(tmp_path: Path) -> None:
    """The compare HTML includes schema diff, drifted features, and at
    least one per-feature card on a clear shift."""
    from conftest import write_two_csvs_with_shift

    from biopsy import compare_profiles
    from biopsy.render.html import render_compare

    a_path, b_path = write_two_csvs_with_shift(tmp_path)
    a = profile(a_path, target="target")
    b = profile(b_path, target="target")
    report = compare_profiles(a, b)
    out = tmp_path / "compare.html"
    rendered = render_compare(a, b, report, out)
    assert rendered.exists()
    text = rendered.read_text()
    assert "Schema diff" in text
    assert "Top drifted features" in text
    assert "feature-card" in text
    assert "age" in text


def test_html_render(tmp_path: Path) -> None:
    csv = write_demo_csv(tmp_path / "demo.csv", n=1000)
    prof = profile(csv, target="churned")
    out = render_html(prof, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text()
    assert "biopsy" in content
    assert "churned" in content
    assert "plotly" in content.lower()
    assert '<script src="https://cdn.plot.ly' not in content
    assert "plotly.js" in content.lower()

    cdn_out = render_html(prof, tmp_path / "report-cdn.html", embed_plotly=False)
    cdn_content = cdn_out.read_text()
    assert '<script src="https://cdn.plot.ly' in cdn_content


