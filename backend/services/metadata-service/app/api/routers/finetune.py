"""
Fine-tuning API using MTMC Ground Truth data.

POST /api/v1/finetune/run
- Load ground truth from MTMC dataset
- Run fine-tuning pipeline
- Return metrics

Future: 40 scenes with videos uploaded directly to temp
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...services.ground_truth_finetune import (
    MTMCFineTuningPipeline,
    MTMCGroundTruthLoader,
    SCENE_CONFIG,
    vinuni_cam_to_scene,
    vinuni_to_scene_camera,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["finetune"])

# Default dataset root
DATASET_ROOT = Path("/workspace/storage/dataset/MTMC_Tracking_2024/train")


class FineTuneRequest(BaseModel):
    """Request to run fine-tuning."""
    scene_name: str = Field(..., description="Scene name, e.g., 'scene_001'")
    camera: Optional[str] = Field(None, description="VinUni camera (e.g., 'cam_01'). None = all cameras")
    segment_ids: Optional[list[int]] = Field(None, description="Segment IDs to process. None = all segments")
    batch_size: int = Field(8, description="Batch size for inference")
    models_to_finetune: list[str] = Field(
        default=["grounding_dino", "eva02"],
        description="Models to fine-tune: grounding_dino, eva02"
    )
    num_samples: Optional[int] = Field(None, description="Max samples to process. None = all")


class FineTuneResponse(BaseModel):
    """Response from fine-tuning run."""
    status: str
    scene_name: str
    camera: Optional[str]
    frames_processed: int
    total_detections: int
    avg_detection_iou: float
    detection_f1: float
    reid_accuracy: float
    embedding_similarity: float
    grounding_dino_loss: float
    eva02_loss: float


class BatchFineTuneRequest(BaseModel):
    """Request to batch fine-tune multiple scenes."""
    scenes: list[str] = Field(..., description="List of scene names")
    models_to_finetune: list[str] = Field(
        default=["grounding_dino", "eva02"],
        description="Models to fine-tune"
    )
    num_samples: Optional[int] = Field(None, description="Max samples per scene")


class GenerateDatasetRequest(BaseModel):
    """Request to generate fine-tuning dataset."""
    scene_name: str = Field(..., description="Scene name")
    output_path: Optional[str] = Field(None, description="Output path")
    num_segments: Optional[int] = Field(None, description="Max segments")


@router.post("/run", response_model=FineTuneResponse)
async def run_finetune(request: FineTuneRequest) -> FineTuneResponse:
    """
    Run fine-tuning on a scene using MTMC ground truth.
    
    Fine-tuning objectives:
    1. Grounding DINO: learn better person detection from GT bboxes
    2. EVA-02: learn better Re-ID embeddings from cross-camera pairs
    """
    logger.info(
        f"Fine-tuning request: scene={request.scene_name}, "
        f"camera={request.camera}, models={request.models_to_finetune}"
    )
    
    # Validate scene
    if request.scene_name not in SCENE_CONFIG:
        raise HTTPException(
            400,
            f"Scene '{request.scene_name}' not found. "
            f"Available: {list(SCENE_CONFIG.keys())}"
        )
    
    # Validate camera if specified
    if request.camera:
        try:
            scene_name, cam_id = vinuni_cam_to_scene(request.camera)
            if scene_name != request.scene_name:
                raise HTTPException(
                    400,
                    f"Camera {request.camera} belongs to scene {scene_name}, "
                    f"not {request.scene_name}"
                )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    
    try:
        # Initialize pipeline
        pipeline = MTMCFineTuningPipeline(
            scene_name=request.scene_name,
            dataset_root=str(DATASET_ROOT),
            batch_size=request.batch_size,
        )
        
        # Run fine-tuning
        metrics = await pipeline.run(
            camera=request.camera,
            segment_ids=request.segment_ids,
            models_to_finetune=request.models_to_finetune,
            num_samples=request.num_samples,
        )
        
        return FineTuneResponse(
            status="completed",
            scene_name=request.scene_name,
            camera=request.camera,
            frames_processed=metrics.frames_processed,
            total_detections=metrics.total_detections,
            avg_detection_iou=metrics.avg_detection_iou,
            detection_f1=metrics.detection_f1,
            reid_accuracy=metrics.reid_accuracy,
            embedding_similarity=metrics.embedding_similarity,
            grounding_dino_loss=metrics.grounding_dino_loss,
            eva02_loss=metrics.eva02_loss,
        )
        
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.error(f"Fine-tuning failed: {exc}", exc_info=True)
        raise HTTPException(500, f"Fine-tuning failed: {exc}")


@router.get("/scenes")
async def list_scenes() -> dict:
    """List available scenes for fine-tuning."""
    # Auto-detect scenes from filesystem
    MTMCGroundTruthLoader(DATASET_ROOT)
    
    scenes = []
    for scene_name, config in sorted(SCENE_CONFIG.items()):
        scenes.append({
            "scene_id": scene_name,
            "scene_num": int(scene_name.split("_")[1]),
            "num_segments": len(config),
            "vinuni_cameras": f"cam_{config['vinuni_range'][0]:02d}-{config['vinuni_range'][1]:02d}",
            "num_cal_cameras": config["cal_cameras"],
            "num_gt_cameras": config["gt_cameras"],
        })
    
    return {
        "scenes": scenes,
        "total": len(scenes),
        "total_cameras": sum(s["num_cal_cameras"] for s in scenes),
    }


@router.get("/scenes/{scene_name}")
async def get_scene_info(scene_name: str) -> dict:
    """Get detailed info about a specific scene."""
    if scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    loader = MTMCGroundTruthLoader(DATASET_ROOT)
    stats = loader.get_scene_stats(scene_name)
    
    if "error" in stats:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    config = SCENE_CONFIG[scene_name]
    
    # Generate VinUni camera list
    start, end = config["vinuni_range"]
    vinuni_cameras = [f"cam_{i:02d}" for i in range(start, end + 1)]
    
    return {
        "scene_id": scene_name,
        "scene_num": int(scene_name.split("_")[1]),
        "num_segments": stats["num_segments"],
        "num_cal_cameras": stats["num_cal_cameras"],
        "num_gt_cameras": stats["num_gt_cameras"],
        "num_vinuni_cameras": stats["num_vinuni_cameras"],
        "num_tracks": stats["num_tracks"],
        "num_detections": stats["num_detections"],
        "segment_range": stats["segment_range"],
        "vinuni_cameras": vinuni_cameras,
        "calibration_cameras": config["cal_cameras"],
    }


@router.get("/scenes/{scene_name}/segments")
async def get_segment_list(scene_name: str) -> dict:
    """Get list of segments in a scene."""
    if scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    loader = MTMCGroundTruthLoader(DATASET_ROOT)
    stats = loader.get_scene_stats(scene_name)
    
    segments = []
    for seg_id in stats.get("segment_range", []):
        seg_stats = loader.get_segment_stats(scene_name, seg_id)
        segments.append({
            "segment_id": seg_id,
            "num_cameras": seg_stats.get("num_cameras", 0),
            "num_video_cameras": seg_stats.get("num_video_cameras", 0),
            "num_tracks": seg_stats.get("num_tracks", 0),
            "num_detections": seg_stats.get("num_detections", 0),
            "video_cameras": seg_stats.get("video_cameras", []),
        })
    
    return {
        "scene_id": scene_name,
        "segments": segments,
        "total_segments": len(segments),
    }


@router.get("/scenes/{scene_name}/segment/{segment_id}")
async def get_segment_info(scene_name: str, segment_id: int) -> dict:
    """Get detailed info about a specific segment."""
    if scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    loader = MTMCGroundTruthLoader(DATASET_ROOT)
    stats = loader.get_segment_stats(scene_name, segment_id)
    
    if "error" in stats:
        raise HTTPException(404, f"Segment not found: {segment_id}")
    
    return {
        "scene_id": scene_name,
        "segment_id": segment_id,
        **stats,
    }


@router.get("/scenes/{scene_name}/track/{track_id}")
async def get_track_trajectory(
    scene_name: str,
    track_id: int,
    segment_id: Optional[int] = None,
) -> dict:
    """
    Get trajectory for a specific track across cameras.
    
    Useful for:
    - Visualizing track patterns
    - Fine-tuning appearance models
    - Analyzing cross-camera matching
    """
    if scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    loader = MTMCGroundTruthLoader(DATASET_ROOT)
    stats = loader.get_scene_stats(scene_name)
    
    if "error" in stats:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    # Get segment IDs to search
    segment_ids = [segment_id] if segment_id else stats.get("segment_range", [])
    
    # Collect all appearances of this track
    appearances = []
    
    for seg_id in segment_ids:
        treklet = loader.get_track_cross_camera(
            scene_name, seg_id, track_id, video_cameras_only=False
        )
        
        for det in treklet.detections:
            vinuni_cam = det.vinuni_cam
            
            appearances.append({
                "segment_id": det.segment_id,
                "camera_id": det.camera_id,
                "vinuni_camera": vinuni_cam,
                "track_id": det.track_id,
                "bbox": det.bbox.to_dict(),
                "world_position": {
                    "x": det.world_pos[0],
                    "y": det.world_pos[1],
                    "z": det.world_pos[2],
                },
            })
    
    if not appearances:
        raise HTTPException(404, f"Track {track_id} not found in scene {scene_name}")
    
    # Group by camera
    camera_appearances: dict[str, list] = {}
    for app in appearances:
        cam = app["vinuni_camera"] or f"camera_{app['camera_id']}"
        if cam not in camera_appearances:
            camera_appearances[cam] = []
        camera_appearances[cam].append(app)
    
    # Get unique cameras
    cameras = list(camera_appearances.keys())
    
    # Compute 3D trajectory
    trajectory_3d = [
        {"segment_id": app["segment_id"], "position": app["world_position"]}
        for app in appearances
    ]
    
    return {
        "track_id": track_id,
        "scene_id": scene_name,
        "num_appearances": len(appearances),
        "num_cameras": len(cameras),
        "cameras": cameras,
        "camera_appearances": camera_appearances,
        "trajectory_3d": trajectory_3d,
    }


@router.get("/scenes/{scene_name}/cross-camera-pairs")
async def get_cross_camera_pairs(
    scene_name: str,
    segment_id: Optional[int] = None,
    min_cameras: int = Query(default=2, description="Minimum cameras that must see the track"),
) -> dict:
    """
    Get cross-camera pairs for Re-ID training.
    
    Returns pairs of cameras that see the same track.
    """
    if scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    loader = MTMCGroundTruthLoader(DATASET_ROOT)
    stats = loader.get_scene_stats(scene_name)
    
    if "error" in stats:
        raise HTTPException(404, f"Scene not found: {scene_name}")
    
    segment_ids = [segment_id] if segment_id else stats.get("segment_range", [])
    
    all_pairs = []
    track_count = 0
    
    for seg_id in segment_ids:
        # Group detections by track_id
        track_groups: dict[int, list] = {}
        
        dets = loader.get_detections(scene_name, seg_id, video_cameras_only=True)
        
        for det in dets:
            if det.track_id not in track_groups:
                track_groups[det.track_id] = []
            track_groups[det.track_id].append(det)
        
        # Generate pairs
        for track_id, dets in track_groups.items():
            cameras = [d.vinuni_cam for d in dets if d.vinuni_cam]
            cameras = sorted(set(cameras))
            
            if len(cameras) < min_cameras:
                continue
            
            # All pairs of cameras
            pairs = []
            for i in range(len(cameras)):
                for j in range(i + 1, len(cameras)):
                    pairs.append([cameras[i], cameras[j]])
            
            all_pairs.append({
                "track_id": track_id,
                "segment_id": seg_id,
                "cameras": cameras,
                "num_cameras": len(cameras),
                "pairs": pairs,
            })
            track_count += 1
    
    return {
        "scene_id": scene_name,
        "segment_ids": segment_ids,
        "total_tracks": track_count,
        "total_pairs": sum(len(p["pairs"]) for p in all_pairs),
        "pairs": all_pairs[:1000],  # Limit response size
    }


@router.post("/batch")
async def batch_finetune(request: BatchFineTuneRequest) -> dict:
    """
    Run fine-tuning on multiple scenes in batch.
    """
    results = []
    
    for scene_name in request.scenes:
        logger.info(f"Batch: Processing {scene_name}...")
        
        req = FineTuneRequest(
            scene_name=scene_name,
            models_to_finetune=request.models_to_finetune,
            num_samples=request.num_samples,
        )
        
        try:
            result = await run_finetune(req)
            results.append({
                "scene_name": scene_name,
                "status": "success",
                "frames_processed": result.frames_processed,
                "total_detections": result.total_detections,
                "detection_f1": result.detection_f1,
                "reid_accuracy": result.reid_accuracy,
            })
        except Exception as exc:
            logger.error(f"Failed to process {scene_name}: {exc}")
            results.append({
                "scene_name": scene_name,
                "status": "failed",
                "error": str(exc),
            })
    
    return {
        "total_scenes": len(request.scenes),
        "completed": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": results,
    }


@router.post("/generate-dataset")
async def generate_finetune_dataset(request: GenerateDatasetRequest) -> dict:
    """
    Generate fine-tuning dataset from ground truth.
    
    Saves cross-camera pairs and detections to JSON for later training.
    """
    from ...services.ground_truth_finetune import generate_finetune_dataset
    
    if request.scene_name not in SCENE_CONFIG:
        raise HTTPException(404, f"Scene not found: {request.scene_name}")
    
    output_path = Path(request.output_path) if request.output_path else DATASET_ROOT.parent / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = generate_finetune_dataset(
            DATASET_ROOT, request.scene_name, output_path, request.num_segments
        )
        
        return {
            "status": "success",
            "scene_name": request.scene_name,
            "output_path": str(output_path / f"{request.scene_name}_finetune.json"),
            "cross_camera_pairs": len(dataset["cross_camera_pairs"]),
            "total_detections": len(dataset["detections"]),
            "num_segments": dataset["num_segments"],
        }
    except Exception as exc:
        logger.error(f"Failed to generate dataset: {exc}")
        raise HTTPException(500, f"Failed to generate dataset: {exc}")
