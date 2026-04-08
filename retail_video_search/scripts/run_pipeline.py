#!/usr/bin/env python3
"""
One-click script to run the entire pipeline:
1. Download datasets (optional)
2. Convert datasets to unified format
3. Index videos into Qdrant
4. Launch Streamlit app
"""
import sys
import subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import logger

def run_command(cmd, description):
    logger.info(f"Running: {description}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download datasets first")
    parser.add_argument("--convert-only", action="store_true", help="Only convert datasets")
    parser.add_argument("--index-only", action="store_true", help="Only index videos")
    parser.add_argument("--ui-only", action="store_true", help="Only launch UI")
    args = parser.parse_args()
    
    # Step 1: Download datasets
    if args.download:
        if not run_command("python scripts/download_datasets.py --all", "Downloading datasets"):
            return
    
    # Step 2: Convert and index
    if not args.ui_only:
        cmd = "python src/pipeline/index_videos.py"
        if args.convert_only:
            cmd += " --convert-only"
        elif args.index_only:
            cmd += " --index-only"
        if not run_command(cmd, "Converting and indexing"):
            return
    
    # Step 3: Launch UI
    if not args.convert_only and not args.index_only:
        run_command("streamlit run src/app.py", "Launching Streamlit UI")

if __name__ == "__main__":
    main()
