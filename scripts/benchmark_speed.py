import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LenslessReconstructionModel


METHODS = {
    "ADMM100": (LenslessReconstructionModel, dict(method="admm100", admm_iterations=100, trainable_admm=False)),
    "LeADMM20": (LenslessReconstructionModel, dict(method="leadmm20", admm_iterations=20, trainable_admm=True)),
    "Pre4+LeADMM5+Post4": (
        LenslessReconstructionModel,
        dict(
            method="modular_pre_post",
            admm_iterations=5,
            trainable_admm=True,
            preprocessor="pre4",
            postprocessor="post4",
        ),
    ),
    "Pre8+LeADMM5": (
        LenslessReconstructionModel,
        dict(
            method="modular_pre_only",
            admm_iterations=5,
            trainable_admm=True,
            preprocessor="pre8",
        ),
    ),
    "LeADMM5+Post8": (
        LenslessReconstructionModel,
        dict(
            method="modular_post_only",
            admm_iterations=5,
            trainable_admm=True,
            postprocessor="post8",
        ),
    ),
    "FISTA": (LenslessReconstructionModel, dict(method="fista", fista_iterations=100)),
}


def measure(model, measurement, psf, warmup, trials):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(measurement=measurement, psf=psf)
        if measurement.is_cuda:
            torch.cuda.synchronize()
        times = []
        for _ in range(trials):
            start = time.perf_counter()
            model(measurement=measurement, psf=psf)
            if measurement.is_cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    tensor = torch.tensor(times)
    return tensor.mean().item(), tensor.std(unbiased=False).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument("--width", type=int, default=507)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    measurement = torch.rand(1, 3, args.height, args.width, device=args.device)
    psf = torch.rand(1, 3, args.height, args.width, device=args.device)
    psf = psf / psf.norm()
    print("method,mean_ms,std_ms")
    for name, (model_cls, kwargs) in METHODS.items():
        model = model_cls(**kwargs).to(args.device)
        mean, std = measure(model, measurement, psf, args.warmup, args.trials)
        print(f"{name},{mean:.3f},{std:.3f}")


if __name__ == "__main__":
    main()
