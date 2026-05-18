"""Tests for the batch CLI."""

from __future__ import annotations

import zipfile
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from unittest.mock import patch

from click.testing import CliRunner

from inspect_otel.cli import cli
from tests.conftest import make_eval_log, make_sample

if TYPE_CHECKING:
    from inspect_ai.log._log import EvalLog


def _write_eval_file(path: Path, log: EvalLog, name: str = "test_log.eval") -> Path:
    eval_path = path / name
    with zipfile.ZipFile(eval_path, "w") as zf:
        header = log.model_dump_json(exclude_none=True, exclude={"samples", "reductions"})
        zf.writestr("header.json", header)
    return eval_path


def test_export_single_file(tmp_path):
    log = make_eval_log(samples=[make_sample()])
    eval_path = _write_eval_file(tmp_path, log)

    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(eval_path), "--dry-run"])
    assert result.exit_code == 0


def test_export_directory(tmp_path):
    log = make_eval_log(samples=[make_sample()])
    _write_eval_file(tmp_path, log)

    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0


def test_dry_run_succeeds_without_network(tmp_path):
    log = make_eval_log(samples=[make_sample()])
    _write_eval_file(tmp_path, log)

    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "Exported 1 file(s)." in result.output


def test_export_no_eval_files(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0


def test_export_with_endpoint(tmp_path, monkeypatch):
    log = make_eval_log(samples=[make_sample()])
    _write_eval_file(tmp_path, log)

    runner = CliRunner()
    with patch("inspect_otel.cli._export_with_otlp") as mock_export:
        result = runner.invoke(cli, ["export", str(tmp_path), "--endpoint", "https://otel.example.com"])
        assert result.exit_code == 0
        mock_export.assert_called_once()


def test_export_latest_only(tmp_path):
    log_a_old = make_eval_log(task="task_a", samples=[make_sample()])
    log_a_old.stats.started_at = "2024-01-01T10:00:00+00:00"
    log_a_new = make_eval_log(task="task_a", samples=[make_sample()])
    log_a_new.stats.started_at = "2024-06-01T10:00:00+00:00"
    log_b = make_eval_log(task="task_b", samples=[make_sample()])
    log_b.stats.started_at = "2024-03-01T10:00:00+00:00"

    _write_eval_file(tmp_path, log_a_old, name="a_old.eval")
    _write_eval_file(tmp_path, log_a_new, name="a_new.eval")
    _write_eval_file(tmp_path, log_b, name="b.eval")

    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(tmp_path), "--dry-run", "--latest-only"])
    assert result.exit_code == 0
    assert "Exported 2 file(s)." in result.output
