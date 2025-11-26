import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class EpochRecord:
    epoch: int
    recon_loss: float
    kl_loss: float
    kl_weight: float


class TrainingMonitor:
    """Utility for recording MT-OBVAE training curves and exporting artifacts."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.records: List[EpochRecord] = []
        os.makedirs(self.output_dir, exist_ok=True)

    def log_epoch(self, epoch: int, recon_loss: float, kl_loss: float, kl_weight: float) -> None:
        self.records.append(EpochRecord(epoch=epoch, recon_loss=recon_loss, kl_loss=kl_loss, kl_weight=kl_weight))

    def _to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([record.__dict__ for record in self.records])

    def save_logs(self) -> None:
        df = self._to_dataframe()
        csv_path = os.path.join(self.output_dir, "training_log.csv")
        json_path = os.path.join(self.output_dir, "training_log.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)

    def plot_curves(self, figsize=(8, 5)) -> None:
        if not self.records:
            return

        df = self._to_dataframe()
        nature_palette = ["#E64B35", "#4DBBD5", "#00A087"]
        fig, ax_loss = plt.subplots(figsize=figsize)

        ax_loss.plot(
            df["epoch"],
            df["recon_loss"],
            label="Reconstruction Loss",
            marker="o",
            color=nature_palette[0],
        )
        ax_loss.plot(
            df["epoch"],
            df["kl_loss"],
            label="KL Loss",
            marker="o",
            color=nature_palette[1],
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")

        ax_weight = ax_loss.twinx()
        ax_weight.plot(
            df["epoch"],
            df["kl_weight"],
            label=r"$\lambda_{t}$",
            linestyle="--",
            color=nature_palette[2],
        )
        ax_weight.set_ylabel(r"$\lambda_{t}$")
        ax_weight.set_ylim(0, 0.02)

        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_weight, labels_weight = ax_weight.get_legend_handles_labels()
        ax_loss.legend(lines_loss + lines_weight, labels_loss + labels_weight, loc="best")

        ax_loss.set_title("MT-OBVAE Training Progress")
        ax_loss.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        fig_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)

    def export(self) -> None:
        self.save_logs()
        self.plot_curves()
