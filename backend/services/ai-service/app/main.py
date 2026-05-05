import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .detection import Detector
from .tracking import Tracker
from .features import EmbeddingExtractor
from .video import VideoSampler
from .core.types.frame import ProcessingResult, VideoMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Service", version="1.0.0")

detector = Detector(
    model_name=settings.detection_model,
    confidence_threshold=settings.detection_confidence_threshold,
)

tracker = Tracker(
    tracking_method="ocsort",
    max_age=30,
    iou_threshold=0.3,
)

embedding_extractor = EmbeddingExtractor(
    model_name=settings.embedding_model,
)

sampler = VideoSampler(
    target_fps=settings.sample_fps,
    max_frames=settings.max_frames_per_video,
)


class ProcessVideoRequest(BaseModel):
    video_path: str
    video_id: Optional[str] = None
    camera_id: Optional[str] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


class PersonCandidate(BaseModel):
    candidate_id: str
    track_id: int
    video_id: str
    camera_id: Optional[str] = None
    frame_idx: int
    bbox: List[float]
    confidence: float
    appearance_embedding: List[float]
    tracklet_feature_pipeline: dict


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-service"}


@app.post("/api/v1/video/process")
async def process_video(request: ProcessVideoRequest):
    start_time = time.time()

    try:
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

        video_info = sampler.get_video_info(video_path)
        if video_info["total_frames"] == 0:
            raise HTTPException(status_code=400, detail="Could not read video")

        frames, frame_indices = sampler.sample_video(
            video_path,
            start_frame=request.start_frame,
            end_frame=request.end_frame,
        )

        if not frames:
            raise HTTPException(status_code=400, detail="No frames sampled from video")

        tracker.reset()
        all_detections = []

        for i, frame in enumerate(frames):
            frame_idx = frame_indices[i]
            timestamp = frame_idx / video_info["fps"]
            detections = detector.detect_persons(frame, frame_idx, timestamp)
            all_detections.append(detections)

        tracklets = tracker.update(all_detections)

        video_metadata = VideoMetadata(
            video_id=request.video_id or video_path.stem,
            total_frames=video_info["total_frames"],
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
            duration_seconds=video_info["duration"],
            camera_id=request.camera_id,
            source_path=str(video_path),
        )

        embeddings = embedding_extractor.extract_batch_embeddings(tracklets, frames)
        for tracklet, emb in zip(tracklets, embeddings):
            if emb is not None:
                tracklet.appearance_embedding = emb

        result = ProcessingResult(
            video_metadata=video_metadata,
            tracklets=tracklets,
            processing_time_seconds=time.time() - start_time,
            model_info={
                "detector": detector.model_info,
                "embedding": embedding_extractor.model_info,
            },
        )

        candidates = []
        for tracklet in tracklets:
            if not tracklet.detections:
                continue

            bbox = tracklet.representative_bbox
            if bbox is None:
                continue

            candidate = PersonCandidate(
                candidate_id=f"{video_metadata.video_id}_track_{tracklet.track_id}",
                track_id=tracklet.track_id,
                video_id=video_metadata.video_id,
                camera_id=request.camera_id,
                frame_idx=tracklet.start_frame,
                bbox=bbox.to_list(),
                confidence=bbox.confidence,
                appearance_embedding=tracklet.appearance_embedding.tolist() if tracklet.appearance_embedding is not None else [],
                tracklet_feature_pipeline={
                    "total_frames": tracklet.duration_frames,
                    "start_frame": tracklet.start_frame,
                    "end_frame": tracklet.end_frame,
                },
            )
            candidates.append(candidate)

        return JSONResponse(content={
            "video": {
                "video_id": video_metadata.video_id,
                "camera_id": video_metadata.camera_id,
                "total_frames": video_metadata.total_frames,
                "fps": video_metadata.fps,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "duration": video_metadata.duration_seconds,
            },
            "person_count": len(candidates),
            "candidates": [c.model_dump() for c in candidates],
            "processing_time": result.processing_time_seconds,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing video")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect")
async def detect_objects(video_path: str, frame_idx: int = 0):
    try:
        frame = sampler.get_frame_at_index(video_path, frame_idx)
        if frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        detections = detector.detect_persons(frame, frame_idx)

        return {
            "frame_idx": frame_idx,
            "detections": [
                {
                    "bbox": d.bbox.to_list(),
                    "confidence": d.bbox.confidence,
                    "class_name": d.bbox.class_name,
                }
                for d in detections
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in detection")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/video/info")
async def get_video_info(video_path: str):
    info = sampler.get_video_info(video_path)
    if info["total_frames"] == 0:
        raise HTTPException(status_code=404, detail="Could not read video")
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
