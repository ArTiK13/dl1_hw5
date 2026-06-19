import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lensless.preprocessor import prepare_measurement, prepare_target
from src.utils.io_utils import extract_roi, load_rgb_tensor


DEFAULT_DATASET_NAME = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
DEFAULT_CACHE_DIR = "data/cache/digicam/hf"
DEFAULT_SAVED_ROOT = "data/saved"
DEFAULT_INDICES = [0, 42, 256, 528]

METHODS = [
    ("admm100", "ADMM100", "admm100_test"),
    ("fista100", "FISTA100", "fista_test"),
    ("leadmm20", "LeADMM20", "leadmm20_test"),
    ("pre8_leadmm5", "Pre8+LeADMM5", "modular_pre_only_test"),
    ("leadmm5_post8", "LeADMM5+Post8", "modular_post_only_test"),
    ("pre4_leadmm5_post4", "Pre4+LeADMM5+Post4", "modular_pre_post_test"),
]

def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def method_by_key():
    return {key: {"label": label, "dir": method_dir} for key, label, method_dir in METHODS}


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape[0] in (1, 3):
        tensor = tensor.permute(1, 2, 0)
    return tensor.numpy()


def psnr(prediction, target):
    mse = F.mse_loss(extract_roi(prediction).clamp(0.0, 1.0), extract_roi(target).clamp(0.0, 1.0))
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def image_id_from_index(index):
    return f"test_{index:06d}"


def load_reference(dataset, index):
    item = dataset[index]
    measurement = prepare_measurement(item["lensless"])
    target = prepare_target(item["lensed"], measurement.shape)
    return measurement, target


def load_prediction(saved_root, method_dir, image_id):
    return load_rgb_tensor(saved_root / method_dir / "test" / f"{image_id}.png")


def load_report_samples(
    indices=DEFAULT_INDICES,
    dataset_name=DEFAULT_DATASET_NAME,
    cache_dir=DEFAULT_CACHE_DIR,
    saved_root=DEFAULT_SAVED_ROOT,
):
    dataset = load_dataset(dataset_name, split="test", cache_dir=project_path(cache_dir))
    saved_root = project_path(saved_root)
    methods = method_by_key()
    samples = []
    for index in indices:
        image_id = image_id_from_index(index)
        measurement, target = load_reference(dataset, index)
        predictions = {
            key: load_prediction(saved_root, method["dir"], image_id)
            for key, method in methods.items()
        }
        metrics = {
            key: psnr(prediction, target)
            for key, prediction in predictions.items()
        }
        samples.append(
            {
                "image_id": image_id,
                "measurement": measurement,
                "target": target,
                "predictions": predictions,
                "metrics": metrics,
            }
        )
    return samples


def add_image(ax, image, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def make_all_methods_figure(samples):
    methods = method_by_key()
    columns = [("target", "Target")] + [(key, method["label"]) for key, method in methods.items()]
    fig, axes = plt.subplots(len(samples), len(columns), figsize=(2.7 * len(columns), 2.4 * len(samples)))
    axes = np.atleast_2d(axes)
    for row, sample in enumerate(samples):
        for col, (key, label) in enumerate(columns):
            ax = axes[row, col]
            if key == "target":
                image = tensor_to_image(extract_roi(sample["target"]))
                title = f"{sample['image_id']}\n{label}"
            else:
                image = tensor_to_image(extract_roi(sample["predictions"][key]))
                title = f"{label}\n{sample['metrics'][key]:.2f}"
            add_image(ax, image, title)
    fig.tight_layout()
    return fig


def show_all_methods_comparison(samples):
    fig = make_all_methods_figure(samples)
    plt.show()
    plt.close(fig)


def selected_metrics_rows(samples):
    methods = method_by_key()
    rows = []
    for sample in samples:
        for key, method in methods.items():
            rows.append(
                {
                    "image_id": sample["image_id"],
                    "method": method["label"],
                    "psnr": f"{sample['metrics'][key]:.6f}",
                }
            )
    return rows


def show_selected_sample_psnr(samples):
    print_table(selected_metrics_rows(samples))


def print_table(rows):
    headers = list(rows[0].keys())
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def write_selected_metrics(samples, output_path):
    output_path = Path(output_path)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "method", "psnr"])
        writer.writeheader()
        writer.writerows(selected_metrics_rows(samples))


def save_report_assets(
    output_dir="analysis/report_assets",
    indices=DEFAULT_INDICES,
    dataset_name=DEFAULT_DATASET_NAME,
    cache_dir=DEFAULT_CACHE_DIR,
    saved_root=DEFAULT_SAVED_ROOT,
):
    output_dir = project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_report_samples(indices, dataset_name, cache_dir, saved_root)

    fig = make_all_methods_figure(samples)
    fig.savefig(output_dir / "qualitative_all_methods.png", dpi=180)
    plt.close(fig)

    for sample in samples:
        fig = make_all_methods_figure([sample])
        fig.savefig(output_dir / f"{sample['image_id']}_all_methods.png", dpi=180)
        plt.close(fig)

    write_selected_metrics(samples, output_dir / "selected_sample_psnr.csv")
