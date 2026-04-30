from __future__ import annotations

from copy import deepcopy


STRICT_PIPELINE: dict = {
    "summary": "Strict single pipeline: RF-DETR + OCMCTrack + SOLIDER+KPR + ITSELF + TrackEval HOTA.",
    "validated_on": "2026-04-25",
    "components": {
        "detector": {
            "name": "RF-DETR 2x-large",
            "official_class_name": "RFDETR2XLarge",
            "official_inference_alias": "rfdetr-2xlarge",
            "official_package": "rfdetr[plus]",
            "family": "detection_transformer",
            "nms_free": True,
            "predict_api": "model.predict(image, threshold=...)",
            "deployment": "TensorRT FP16",
            "resolution": "880x880",
            "sources": [
                {"label": "ICLR 2026 poster", "url": "https://iclr.cc/virtual/2026/poster/10007257"},
                {"label": "roboflow/rf-detr", "url": "https://github.com/roboflow/rf-detr"},
            ],
        },
        "tracker": {
            "name": "OCMCTrack-style corrective cascade",
            "family": "online_mtmc",
            "geometry_aware": True,
            "occlusion_reasoning": True,
            "sources": [
                {
                    "label": "OCMCTrack CVPRW 2024",
                    "url": "https://publica.fraunhofer.de/entities/publication/b8475da1-6fa2-4696-8d3d-78323b3d50cf",
                },
                {
                    "label": "NVIDIA Sparse4D docs",
                    "url": "https://docs.nvidia.com/vss/3.1.0/warehouse-docs/Sparse4D.html",
                },
            ],
        },
        "reid": {
            "name": "SOLIDER + KPR",
            "family": "human_foundation_reid",
            "part_based": True,
            "occlusion_robust": True,
            "sources": [
                {"label": "SOLIDER", "url": "https://github.com/tinyvision/SOLIDER"},
                {
                    "label": "KPR ECCV 2024",
                    "url": "https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10180_ECCV_2024_paper.php",
                },
            ],
        },
        "semantic_search": {
            "name": "ITSELF",
            "family": "vision_language_retrieval",
            "fine_grained_alignment": True,
            "sources": [
                {
                    "label": "ITSELF WACV 2026",
                    "url": "https://openaccess.thecvf.com/content/WACV2026/html/Nguyen_ITSELF_Attention_Guided_Fine-Grained_Alignment_for_Vision-Language_Retrieval_WACV_2026_paper.html",
                }
            ],
        },
        "evaluation": {
            "name": "TrackEval HOTA",
            "family": "tracking_evaluation",
            "primary_metrics": ["HOTA"],
            "sources": [{"label": "TrackEval", "url": "https://github.com/JonathonLuiten/TrackEval"}],
        },
    },
    "runtime_defaults": {
        "detector_precision": "fp16",
        "association_mode": "geometry_then_reid_then_corrective_cascade",
        "search_mode": "itself_attention_guided",
    },
    "hyperparameters": {
        "detector": {"confidence_threshold": 0.32, "person_class_only": True, "max_queries": 300},
        "tracker": {
            "high_confidence_threshold": 0.45,
            "low_confidence_threshold": 0.12,
            "new_track_threshold": 0.55,
            "iou_gate": 0.18,
            "appearance_gate": 0.22,
            "corrective_buffer_seconds": 14.0,
            "world_gate_max_speed_mps": 2.8,
        },
        "reid": {
            "global_embedding_weight": 0.55,
            "part_embedding_weight": 0.45,
            "cross_camera_match_threshold": 0.27,
        },
        "semantic_search": {
            "fetch_multiplier": 10,
            "topk": 10,
            "score_weights": {"embedding": 0.58, "semantic_overlap": 0.22, "visibility": 0.12, "world_position": 0.08},
        },
    },
}


def get_strict_pipeline() -> dict:
    return deepcopy(STRICT_PIPELINE)
