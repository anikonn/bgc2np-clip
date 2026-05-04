from __future__ import annotations

import numpy as np
import torch


def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return a confusion matrix with rows=true classes and columns=predicted classes."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    true_cpu = y_true.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    pred_cpu = y_pred.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if true_cpu.numel() != pred_cpu.numel():
        raise ValueError("y_true and y_pred must have the same number of elements.")
    if true_cpu.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)

    valid = (true_cpu >= 0) & (true_cpu < num_classes) & (pred_cpu >= 0) & (pred_cpu < num_classes)
    if not bool(valid.all()):
        raise ValueError("y_true and y_pred must contain class indices in [0, num_classes).")

    flat_idx = true_cpu * num_classes + pred_cpu
    cm = torch.bincount(flat_idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).to(dtype=torch.long)


def confusion_matrix_normalized(cm: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize a confusion matrix by true-class rows or predicted-class columns."""
    cm_float = cm.detach().to(dtype=torch.float32, device="cpu")
    if mode == "true":
        denom = cm_float.sum(dim=1, keepdim=True)
    elif mode == "pred":
        denom = cm_float.sum(dim=0, keepdim=True)
    else:
        raise ValueError("mode must be 'true' or 'pred'.")
    return torch.where(denom > 0, cm_float / denom.clamp_min(1.0), torch.zeros_like(cm_float))


def per_class_prf(cm: torch.Tensor) -> dict[str, list[float]]:
    """Compute per-class precision, recall, F1, and support from a confusion matrix."""
    cm_float = cm.detach().to(dtype=torch.float64, device="cpu")
    true_pos = torch.diag(cm_float)
    pred_count = cm_float.sum(dim=0)
    true_count = cm_float.sum(dim=1)

    precision = torch.where(pred_count > 0, true_pos / pred_count.clamp_min(1.0), torch.zeros_like(true_pos))
    recall = torch.where(true_count > 0, true_pos / true_count.clamp_min(1.0), torch.zeros_like(true_pos))
    denom = precision + recall
    f1 = torch.where(denom > 0, (2.0 * precision * recall) / denom.clamp_min(1e-12), torch.zeros_like(denom))

    return {
        "precision": [float(x) for x in precision.tolist()],
        "recall": [float(x) for x in recall.tolist()],
        "f1": [float(x) for x in f1.tolist()],
        "support": [float(x) for x in true_count.tolist()],
    }


def macro_micro_f1_from_cm(cm: torch.Tensor) -> dict[str, float]:
    """Return macro-F1, micro-F1, and accuracy from a confusion matrix."""
    per_class = per_class_prf(cm)
    f1_values = per_class["f1"]
    macro_f1 = float(sum(f1_values) / max(len(f1_values), 1))

    cm_float = cm.detach().to(dtype=torch.float64, device="cpu")
    true_pos = float(torch.diag(cm_float).sum().item())
    total = float(cm_float.sum().item())
    micro_f1 = 0.0 if total == 0.0 else true_pos / total
    accuracy = micro_f1
    return {"macro_f1": macro_f1, "micro_f1": float(micro_f1), "accuracy": float(accuracy)}


def _class_keys(num_classes: int, class_names: list[str] | None) -> list[str]:
    if class_names is None:
        return [str(idx) for idx in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match num_classes.")
    return [str(name) for name in class_names]


def wrong_class_ratios(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Return true-class and predicted-class distributions among wrong predictions."""
    keys = _class_keys(num_classes, class_names)
    true_cpu = y_true.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    pred_cpu = y_pred.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if true_cpu.numel() != pred_cpu.numel():
        raise ValueError("y_true and y_pred must have the same number of elements.")

    wrong_mask = true_cpu != pred_cpu
    wrong_true = true_cpu[wrong_mask]
    wrong_pred = pred_cpu[wrong_mask]

    def ratio_dict(values: torch.Tensor) -> dict[str, float]:
        counts = torch.bincount(values, minlength=num_classes).to(dtype=torch.float64)
        total = float(counts.sum().item())
        if total == 0.0:
            return {key: 0.0 for key in keys}
        ratios = counts / total
        return {key: float(value) for key, value in zip(keys, ratios.tolist(), strict=True)}

    return {
        "ratio_true_among_wrongs": ratio_dict(wrong_true),
        "ratio_pred_among_wrongs": ratio_dict(wrong_pred),
    }


def _metrics_for_predictions(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    return macro_micro_f1_from_cm(cm)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _summarize_trials(trial_metrics: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("accuracy", "macro_f1", "micro_f1"):
        mean, std = _mean_std([float(metrics[metric_name]) for metrics in trial_metrics])
        summary[f"{metric_name}_mean"] = mean
        summary[f"{metric_name}_std"] = std
    return summary


def random_baselines(
    y_train: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    trials: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate majority, uniform-random, and train-prior-random baselines."""
    if trials <= 0:
        raise ValueError("trials must be positive.")

    train_cpu = y_train.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    true_cpu = y_true.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if train_cpu.numel() == 0:
        raise ValueError("y_train must contain at least one label.")

    train_counts = torch.bincount(train_cpu, minlength=num_classes).to(dtype=torch.float64)
    majority_class = int(torch.argmax(train_counts).item())
    majority_pred = torch.full_like(true_cpu, fill_value=majority_class)
    majority = _metrics_for_predictions(true_cpu, majority_pred, num_classes)
    for metric_name, value in list(majority.items()):
        majority[f"{metric_name}_mean"] = float(value)
        majority[f"{metric_name}_std"] = 0.0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    prior_probs = train_counts / train_counts.sum().clamp_min(1.0)

    uniform_trials: list[dict[str, float]] = []
    prior_trials: list[dict[str, float]] = []
    for _ in range(int(trials)):
        uniform_pred = torch.randint(0, num_classes, true_cpu.shape, generator=generator, dtype=torch.long)
        prior_pred = torch.multinomial(prior_probs, num_samples=true_cpu.numel(), replacement=True, generator=generator)
        uniform_trials.append(_metrics_for_predictions(true_cpu, uniform_pred, num_classes))
        prior_trials.append(_metrics_for_predictions(true_cpu, prior_pred, num_classes))

    return {
        "majority": majority,
        "uniform": _summarize_trials(uniform_trials),
        "prior": _summarize_trials(prior_trials),
    }
