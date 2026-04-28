"""Centralized path configuration."""

from pathlib import Path
from dataclasses import dataclass


PROJECT_ROOT = Path(__file__).resolve().parent
DROPBOX_DATA = Path.home() / "Dropbox" / "RL_data"


@dataclass(frozen=True)
class ProjectPaths:
    root:        Path = PROJECT_ROOT
    train_csv:   Path = DROPBOX_DATA / "RL_Final_Merged_train.csv"
    test_csv:    Path = DROPBOX_DATA / "RL_Final_Merged_test.csv"
    synth_pool:  Path = DROPBOX_DATA / "synth_pool.npz"
    
    # Raw T-Bill rates — economic input to the simulator, not a feature.
    dtb3_csv:    Path = PROJECT_ROOT / "data" / "raw" / "DTB3_StockData_RL.csv"
    
    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    tb_logs:     Path = PROJECT_ROOT / "tb_logs"
    final_models: Path = DROPBOX_DATA / "final_checkpoints"

    def ensure_output_dirs(self):
        for p in [self.checkpoints, self.tb_logs, self.final_models]:
            p.mkdir(parents=True, exist_ok=True)

    def validate_inputs(self) -> dict:
        return {
            "train_csv":  self.train_csv.exists(),
            "test_csv":   self.test_csv.exists(),
            "synth_pool": self.synth_pool.exists(),
            "dtb3_csv":   self.dtb3_csv.exists(),
        }


PATHS = ProjectPaths()


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DROPBOX_DATA: {DROPBOX_DATA}  exists={DROPBOX_DATA.exists()}")
    print()
    print("Configured paths:")
    for name, value in PATHS.__dict__.items():
        if isinstance(value, Path):
            exists = "✓" if value.exists() else "✗"
            print(f"  {exists}  {name:14s} {value}")
