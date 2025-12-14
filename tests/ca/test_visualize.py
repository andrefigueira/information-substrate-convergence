from pathlib import Path
import numpy as np

from ca.visualize import plot_convergence, save_snapshots


def test_visual_outputs(tmp_path: Path):
    scores = [0.5, 0.4, 0.3]
    out_png = tmp_path / "conv.png"
    plot_convergence(scores, out_png)
    assert out_png.exists()

    history = np.random.randint(0, 2, (5, 5, 5), dtype=bool)
    out_dir = tmp_path / "snaps"
    save_snapshots(history, out_dir, per_step=2)
    files = list(out_dir.glob("*.png"))
    assert files
