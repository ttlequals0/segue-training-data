"""Download trained LoRA adapter weights from Tinker and build a merged HF model directory.

Usage:
    uv run python tools/export_model.py --tinker-path "tinker://<run-id>/sampler_weights/final"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tinker-path",
        required=True,
        help="Tinker sampler path from training logs (e.g. tinker://.../sampler_weights/final)",
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
        help=f"Base Hugging Face model ID (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--adapter-dir",
        default=str(REPO_ROOT / ".local" / "adapter"),
        help="Directory to save downloaded LoRA adapter weights",
    )
    ap.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / ".local" / "segue-4b-bf16"),
        help="Directory to write merged Hugging Face model",
    )
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        help="Precision format for merged weights (default: bfloat16)",
    )
    args = ap.parse_args()

    from tinker_cookbook import weights

    print(f"Downloading adapter weights from: {args.tinker_path}")
    print(f"  -> Adapter target: {args.adapter_dir}")
    weights.download(
        tinker_path=args.tinker_path,
        output_dir=args.adapter_dir,
    )

    print(f"\nBuilding merged Hugging Face model...")
    print(f"  -> Base model: {args.base_model}")
    print(f"  -> Output path: {args.output_dir}")
    weights.build_hf_model(
        base_model=args.base_model,
        adapter_path=args.adapter_dir,
        output_path=args.output_dir,
        dtype=args.dtype,
    )

    print(f"\nDone. Exported merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
