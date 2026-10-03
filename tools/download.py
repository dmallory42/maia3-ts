"""Download the Maia-3 5M checkpoint into maia/checkpoints/."""

from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
REPO_ID = "UofTCSSLab/Maia3-5M"
FILENAME = "maia3-5m.pt"


def checkpoint_path() -> Path:
    """Return the local checkpoint path, downloading it if missing."""
    target = HERE / "checkpoints" / FILENAME
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=target.parent)
    return target


if __name__ == "__main__":
    print(checkpoint_path())
