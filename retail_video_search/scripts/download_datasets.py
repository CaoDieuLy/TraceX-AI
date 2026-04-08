"""
Download multiple retail datasets from various sources.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config import config
from src.utils.logger import logger
import subprocess

def download_surveillance_retail():
    """Download Surveillance for Retail Stores from Kaggle."""
    try:
        import kagglehub
        logger.info("Downloading Surveillance for Retail Stores dataset...")
        # The dataset is from a Kaggle competition
        path = kagglehub.competition_download("surveillance-for-retail-stores")
        logger.info(f"Downloaded to: {path}")
        # Copy to raw data folder
        import shutil
        target = config.raw_data_root / "surveillance_retail"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        logger.info(f"Copied to {target}")
    except Exception as e:
        logger.error(f"Failed to download Surveillance dataset: {e}")

def download_physicalai():
    """Download PhysicalAI-SmartSpaces from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
        logger.info("Downloading PhysicalAI-SmartSpaces dataset...")
        # Download only a subset (e.g., first warehouse) to save space
        snapshot_download(
            repo_id="nvidia/PhysicalAI-SmartSpaces",
            repo_type="dataset",
            allow_patterns=["MTMC_Tracking_2025/train/Warehouse_000/*"],
            local_dir=config.raw_data_root / "physicalai_smartspaces",
        )
        logger.info("Download completed")
    except Exception as e:
        logger.error(f"Failed to download PhysicalAI dataset: {e}")

def download_retailaction():
    """Download RetailAction via FiftyOne."""
    try:
        import fiftyone as fo
        from fiftyone.utils.huggingface import load_from_hub
        logger.info("Loading RetailAction dataset via FiftyOne...")
        dataset = load_from_hub("Voxel51/RetailAction", max_samples=10)
        # Dataset is loaded, no need to copy
        logger.info("RetailAction loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load RetailAction: {e}")

def download_mimex():
    """Download MIMEX dataset (if publicly available)."""
    # MIMEX dataset may not be directly downloadable
    logger.info("MIMEX dataset requires manual download. See paper: https://arxiv.org/abs/2409.12345")
    logger.info("Creating placeholder directory...")
    mimex_dir = config.raw_data_root / "mimex"
    mimex_dir.mkdir(parents=True, exist_ok=True)
    with open(mimex_dir / "README.txt", "w") as f:
        f.write("MIMEX dataset for fine-grained retail product classification.\n")
        f.write("Download from: https://github.com/anon/MIMEX (not yet public)\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--surveillance", action="store_true")
    parser.add_argument("--physicalai", action="store_true")
    parser.add_argument("--retailaction", action="store_true")
    parser.add_argument("--mimex", action="store_true")
    args = parser.parse_args()
    
    if args.all or args.surveillance:
        download_surveillance_retail()
    if args.all or args.physicalai:
        download_physicalai()
    if args.all or args.retailaction:
        download_retailaction()
    if args.all or args.mimex:
        download_mimex()
