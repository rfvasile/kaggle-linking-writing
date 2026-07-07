import argparse
import json
from pathlib import Path

ROOT = Path.cwd()


def generate_metadata(folder: Path, competition: str, owner: str, slug: str) -> dict:
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition")
    parser.add_argument("--kernel")
    args = parser.parse_args()

    competition = args.competition.rstrip("/").split("/")[-1]
    kernel = args.kernel.rstrip("/")
    owner, slug, *_ = kernel.split("/")
    if "/" not in kernel:
        raise SystemExit("Kernel must be an owner/notebook-slug handle")

    folder = ROOT / "notebooks" / "competitions" / competition / owner / slug
    if Path(folder / "kernel-metadata.json").exists():
       print("Metadata file already present, not overwriting it")
       return
    generate_metadata(folder, competition, owner, slug)


if __name__ == "__main__":
    main()
