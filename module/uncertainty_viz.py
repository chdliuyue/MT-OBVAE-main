import json
import os
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _get_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def mc_sample_probabilities(model: torch.nn.Module, x_batch: torch.Tensor, num_mc_samples: int = 100) -> Dict[str, np.ndarray]:
    device = _get_device(model)
    task_names = ["ttc", "drac", "psd"]
    x_batch = x_batch.to(device)

    model.eval()
    all_prob_preds = {task: [] for task in task_names}
    with torch.no_grad():
        for _ in range(num_mc_samples):
            outputs = model(x_batch)
            z_sample = outputs["z_sample"]
            for task in task_names:
                beta, thresholds = outputs[f"params_{task}"]
                latent_pred = torch.bmm(z_sample.unsqueeze(1), beta).squeeze(-1).squeeze(-1)
                padded_thresholds = F.pad(thresholds, (1, 1), "constant", torch.inf)
                padded_thresholds[:, 0] = -torch.inf
                cdf_vals = torch.distributions.Normal(0, 1).cdf(padded_thresholds - latent_pred.unsqueeze(1))
                probs = cdf_vals[:, 1:] - cdf_vals[:, :-1]
                all_prob_preds[task].append(probs.cpu().numpy())

    return {task: np.concatenate(samples, axis=0) for task, samples in all_prob_preds.items()}


def summarize_uncertainty(prob_samples: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for task, samples in prob_samples.items():
        mean_prob = samples.mean(axis=0)
        epistemic_var = samples.var(axis=0).sum()

        # Mutual information approximation: H(mean) - mean(H(p_i))
        mean_entropy = -np.sum(mean_prob * np.log(mean_prob + 1e-9))
        entropies = -np.sum(samples * np.log(samples + 1e-9), axis=1)
        expected_entropy = entropies.mean()
        mutual_information = mean_entropy - expected_entropy

        summaries[task] = {
            "predictive_mean": mean_prob.tolist(),
            "epistemic_variance": float(epistemic_var),
            "mutual_information": float(mutual_information),
        }
    return summaries


def plot_probability_distributions(
    prob_samples: Dict[str, np.ndarray], output_dir: str, prefix: str = "3_sample"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for task, samples in prob_samples.items():
        plt.figure(figsize=(7, 4))
        plt.boxplot(samples, labels=["class0", "class1", "class2", "class3"])
        plt.ylabel("Predicted probability")
        plt.title(f"MC predictive distribution - {task}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_{task}_prob_boxplot.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist(samples.flatten(), bins=30, alpha=0.7)
        plt.xlabel("Predicted probability")
        plt.ylabel("Frequency")
        plt.title(f"Probability histogram - {task}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_{task}_prob_hist.png"), dpi=300)
        plt.close()


def visualize_uncertainty_cases(
    model: torch.nn.Module,
    X_data: np.ndarray,
    sample_indices: Sequence[int],
    output_dir: str,
    num_mc_samples: int = 100,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    device = _get_device(model)

    for idx in sample_indices:
        x_tensor = torch.tensor(X_data[idx: idx + 1]).to(device)
        with torch.no_grad():
            _, log_var_z = model.encoder(x_tensor)
            aleatoric_index = float(torch.sum(torch.exp(log_var_z)).item())

        prob_samples = mc_sample_probabilities(model, x_tensor, num_mc_samples=num_mc_samples)
        summaries = summarize_uncertainty(prob_samples)
        results.append({"sample_index": idx, "aleatoric_index": aleatoric_index, "summaries": summaries})

        prefix = f"3_sample_{idx}"
        plot_probability_distributions(prob_samples, output_dir, prefix=prefix)

        json_path = os.path.join(output_dir, f"{prefix}_uncertainty.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"sample_index": idx, "aleatoric_index": aleatoric_index, "summaries": summaries}, f, indent=2, ensure_ascii=False)

    return results
