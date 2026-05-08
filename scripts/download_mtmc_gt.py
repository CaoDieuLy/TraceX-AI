#!/usr/bin/env python3
"""
MTMC Tracking 2024 Dataset Setup & Ground Truth Download.

Flow:
    NVIDIA Source → VinUni Temp → VinUni Storage
    D:\119\...     (Driver)      /path/to/Storage

Scene/Camera Mapping:
    scene_001: camera_0001-0010 (10 cameras) → cam_01-10
    scene_002: camera_0011-0019 (9 cameras)  → cam_11-19
    scene_003: camera_0020-0029 (10 cameras) → cam_20-29
    scene_004: camera_0030-0037 (8 cameras)  → cam_30-37
    scene_005: camera_0038-0047 (10 cameras) → cam_38-47
    scene_006: camera_0048-0055 (8 cameras)  → cam_48-55

Usage:
    # Download GT from HuggingFace
    python download_gt.py --hf-repo NVIDIA/MCMT-MTMC-2024 --output-dir ./dataset
    
    # Or copy from NVIDIA source (if you have it)
    python download_gt.py --source D:\119\MTMC_Tracking_2024 --output-dir ./dataset
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scene to camera range mapping
SCENE_CAMERA_RANGES = {
    "scene_001": (1, 10),    # camera_0001-0010
    "scene_002": (11, 19),   # camera_0011-0019
    "scene_003": (20, 29),   # camera_0020-0029
    "scene_004": (30, 37),   # camera_0030-0037
    "scene_005": (38, 47),   # camera_0038-0047
    "scene_006": (48, 55),   # camera_0048-0055
}

# Reverse: camera_number → (scene_name, camera_in_scene)
CAM_TO_SCENE = {}
for scene_name, (start, end) in SCENE_CAMERA_RANGES.items():
    for cam_num in range(start, end + 1):
        # camera index within scene (0-indexed)
        camera_idx = cam_num - start
        CAM_TO_SCENE[cam_num] = (scene_name, camera_idx)


def vinuni_to_camera_id(vinuni_cam: str) -> str:
    """
    Convert VinUni camera name to GT camera ID.
    
    cam_01 → Camera_0000
    cam_11 → Camera_0000  (scene_002 starts from 0)
    """
    cam_num = int(vinuni_cam.replace("cam_", ""))
    # All GT camera IDs are 0-indexed within their scene
    # So cam_01 and cam_11 both map to Camera_0000 (in different scenes)
    scene_name, camera_idx = CAM_TO_SCENE.get(cam_num, (None, None))
    if scene_name is None:
        raise ValueError(f"Unknown camera: {vinuni_cam}")
    return f"Camera_{camera_idx:04d}"


def camera_id_to_vinuni(camera_id: str, scene_name: str) -> str:
    """
    Convert GT camera ID to VinUni camera name.
    
    Camera_0000 + scene_001 → cam_01
    Camera_0000 + scene_002 → cam_11
    """
    camera_idx = int(camera_id.replace("Camera_", ""))
    scene_num = int(scene_name.split("_")[1])
    
    # Find base cam number for this scene
    base_cam = None
    for sn, (start, _) in SCENE_CAMERA_RANGES.items():
        if sn == scene_name:
            base_cam = start
            break
    
    if base_cam is None:
        raise ValueError(f"Unknown scene: {scene_name}")
    
    return f"cam_{base_cam + camera_idx:02d}"


def download_from_huggingface(
    hf_repo: str,
    output_dir: Path,
    scenes: list[str] | None = None,
):
    """
    Download ground truth from HuggingFace.
    
    Args:
        hf_repo: HuggingFace repository (e.g., "NVIDIA/MCMT-MTMC-2024")
        output_dir: Where to save downloaded files
        scenes: List of scenes to download (None = all)
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("Install huggingface_hub: pip install huggingface_hub")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading from {hf_repo}...")
    logger.info(f"Output: {output_dir}")
    
    # Download entire repo
    repo_path = snapshot_download(
        repo_id=hf_repo,
        local_dir=output_dir,
        allow_patterns=["*ground_truth*", "*calibration*"],
    )
    
    logger.info(f"Downloaded to {repo_path}")
    
    # Copy ground truth files to correct locations
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    for scene_dir in sorted(Path(repo_path).glob("train/scene_*")):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        
        if scenes and scene_name not in scenes:
            continue
        
        target_scene = train_dir / scene_name
        target_scene.mkdir(exist_ok=True)
        
        # Copy ground truth files
        for gt_file in scene_dir.glob("*ground_truth*"):
            shutil.copy2(gt_file, target_scene / gt_file.name)
            logger.info(f"  Copied {gt_file.name} → {scene_name}/")
        
        # Copy calibration files
        for cal_file in scene_dir.glob("*calibration*"):
            shutil.copy2(cal_file, target_scene / cal_file.name)
            logger.info(f"  Copied {cal_file.name} → {scene_name}/")


def copy_from_nvidia_source(
    nvidia_source: Path,
    output_dir: Path,
    scenes: list[str] | None = None,
):
    """
    Copy ground truth from local NVIDIA source.
    
    Args:
        nvidia_source: Path to D:\119\MTMC_Tracking_2024
        output_dir: Where to save copied files
        scenes: List of scenes to copy (None = all)
    """
    nvidia_source = Path(nvidia_source)
    output_dir = Path(output_dir)
    
    train_dir = nvidia_source / "train"
    
    if not train_dir.exists():
        logger.error(f"Not a valid NVIDIA source: {train_dir} not found")
        return
    
    logger.info(f"Copying from {nvidia_source}...")
    logger.info(f"Output: {output_dir}")
    
    output_train = output_dir / "train"
    output_train.mkdir(parents=True, exist_ok=True)
    
    for scene_dir in sorted(train_dir.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        
        if scenes and scene_name not in scenes:
            continue
        
        target_scene = output_train / scene_name
        target_scene.mkdir(exist_ok=True)
        
        # Copy ground truth files
        for gt_file in scene_dir.glob("*ground_truth*"):
            shutil.copy2(gt_file, target_scene / gt_file.name)
            logger.info(f"  Copied {gt_file.name} → {scene_name}/")
        
        # Copy calibration files
        for cal_file in scene_dir.glob("*calibration*"):
            shutil.copy2(cal_file, target_scene / cal_file.name)
            logger.info(f"  Copied {cal_file.name} → {scene_name}/")


def generate_camera_mapping_report(output_dir: Path):
    """Generate a mapping report for all cameras."""
    report = {
        "description": "MTMC Tracking 2024 Camera Mapping Report",
        "vinuni_to_gt": {},
        "gt_to_vinuni": {},
        "scene_cameras": {},
    }
    
    for scene_name, (start, end) in SCENE_CAMERA_RANGES.items():
        scene_cameras = {}
        
        for cam_num in range(start, end + 1):
            scene_name2, camera_idx = CAM_TO_SCENE[cam_num]
            vinuni_cam = f"cam_{cam_num:02d}"
            camera_id = f"Camera_{camera_idx:04d}"
            
            # GT camera folder name
            gt_folder = f"camera_{cam_num:04d}"
            
            report["vinuni_to_gt"][vinuni_cam] = {
                "scene": scene_name,
                "camera_id": camera_id,
                "gt_folder": gt_folder,
            }
            
            report["gt_to_vinuni"][f"{scene_name}/{gt_folder}"] = {
                "vinuni_cam": vinuni_cam,
                "camera_id": camera_id,
            }
            
            scene_cameras[vinuni_cam] = {
                "camera_id": camera_id,
                "gt_folder": gt_folder,
            }
        
        report["scene_cameras"][scene_name] = {
            "camera_range": f"camera_{start:04d}-{end:04d}",
            "vinuni_range": f"cam_{start:02d}-cam_{end:02d}",
            "count": end - start + 1,
            "cameras": scene_cameras,
        }
    
    report_path = output_dir / "camera_mapping.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Mapping report: {report_path}")
    
    # Also print summary
    print("\n" + "="*60)
    print("CAMERA MAPPING SUMMARY")
    print("="*60)
    
    for scene_name, info in report["scene_cameras"].items():
        print(f"\n{scene_name}: {info['camera_range']} → {info['vinuni_range']} ({info['count']} cameras)")
        first_cam = list(info["cameras"].keys())[0]
        last_cam = list(info["cameras"].keys())[-1]
        print(f"  {first_cam} ... {last_cam}")


def main():
    parser = argparse.ArgumentParser(
        description="Download/copy MTMC Tracking 2024 ground truth"
    )
    
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hf-repo",
        help="HuggingFace repository (e.g., NVIDIA/MCMT-MTMC-2024)",
    )
    source.add_argument(
        "--source",
        help="Local NVIDIA source path (e.g., D:\\119\\MTMC_Tracking_2024)",
    )
    
    parser.add_argument(
        "--output-dir",
        default="./dataset/MTMC_Tracking_2024",
        help="Output directory (default: ./dataset/MTMC_Tracking_2024)",
    )
    
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Specific scenes to download (e.g., scene_001 scene_002)",
    )
    
    parser.add_argument(
        "--generate-mapping",
        action="store_true",
        help="Generate camera mapping report",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.hf_repo:
        download_from_huggingface(args.hf_repo, output_dir, args.scenes)
    elif args.source:
        copy_from_nvidia_source(Path(args.source), output_dir, args.scenes)
    
    if args.generate_mapping or args.hf_repo or args.source:
        generate_camera_mapping_report(output_dir)


if __name__ == "__main__":
    main()
