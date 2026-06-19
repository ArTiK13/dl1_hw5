import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "xartik/hw5-lensless-computational-imaging-checkpoints"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATTERNS = [
    "leadmm20.pth",
    "modular_pre_post.pth",
    "modular_pre_only.pth",
    "modular_post_only.pth",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=CHECKPOINT_PATTERNS,
    )
    print(f"Downloaded checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
