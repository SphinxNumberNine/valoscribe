import json
from pathlib import Path

import pytest

from valoscribe.scraper.vlr_scraper import VLRScraper
from valoscribe.validation.pipeline import processing_metadata, run_recorded, sha256

FIXTURES = Path(__file__).parents[1] / "fixtures/pipeline"


def test_real_vlr_snapshot_produces_reviewed_metadata():
    policy = json.loads((FIXTURES / "manifest.json").read_text())
    actual = VLRScraper().parse_match_html(
        (FIXTURES / "vlr.html").read_text(encoding="utf-8"), policy["match_url"]
    )
    assert actual == json.loads((FIXTURES / "series.json").read_text())
    assert len(actual["maps"]) == 5
    assert actual["maps"][0]["vod_url"] == "https://youtu.be/WPm6FMkZ3Qo?t=132"
    assert processing_metadata(actual["maps"][0]) == json.loads(
        (FIXTURES / "metadata.json").read_text()
    )


def test_manifest_hashes_match_assets():
    policy = json.loads((FIXTURES / "manifest.json").read_text())
    for name, expected in policy["sha256"].items():
        assert sha256(FIXTURES / name) == expected


def test_missing_and_tampered_fixtures_fail_before_processing(tmp_path):
    with pytest.raises(OSError):
        run_recorded(tmp_path, tmp_path / "out")
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"vlr.html": "bad"}}))
    (tmp_path / "vlr.html").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        run_recorded(tmp_path, tmp_path / "out")


def test_incomplete_metadata_fails():
    with pytest.raises(ValueError):
        processing_metadata({"teams": []})
