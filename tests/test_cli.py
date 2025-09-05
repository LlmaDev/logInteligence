from autoReport import main
from click.testing import CliRunner
from pathlib import Path
import tempfile

def test_cli_runs(tmp_path):
    runner = CliRunner()
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "Pivo2/01/01").mkdir(parents=True)
    (log_dir / "Pivo2/01/01/MESSAGE.txt").write_text(
        "2025-09-02 15:23:01 -> OK-Farm1-16-45-90-95-123456$"
    )
    result = runner.invoke(main, ["--root", str(tmp_path), "--pivots", "Pivo2"])
    assert result.exit_code == 0
