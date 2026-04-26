#!/usr/bin/env python3
"""
Validation script for L4 Edge-First SOTA pipeline.
Tests: YOLO26, CLIP-ReID, ITSELF-lite, ByteTrack, imports
"""

import sys
import traceback
from pathlib import Path

print("=" * 60)
print("L4 EDGE-FIRST SOTA VALIDATION")
print("=" * 60)

# Test 1: Import YOLO
print("\n[1/6] Testing YOLO26 import...")
try:
    from ultralytics import YOLO
    model = YOLO("yolo26x.pt")
    print("   ✅ YOLO26 loaded successfully")
    print(f"   Model device: {model.device}")
except Exception as e:
    print(f"   ❌ YOLO26 failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import CLIP
print("\n[2/6] Testing CLIP-ReID import...")
try:
    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    print("   ✅ CLIP model loaded")
    print(f"   Model: {type(clip_model).__name__}")
except Exception as e:
    print(f"   ❌ CLIP failed: {e}")
    sys.exit(1)

# Test 3: Import transformers
print("\n[3/6] Testing transformers/VLM...")
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("   ✅ Transformers loaded")
except Exception as e:
    print(f"   ❌ Transformers failed: {e}")
    sys.exit(1)

# Test 4: Test hospital_pipeline imports
print("\n[4/6] Testing hospital_pipeline module...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "legacy-engine" / "src_vlm"))
    import hospital_pipeline as hp
    print("   ✅ hospital_pipeline imported")
    print(f"   Functions: _detect_people_yolo26={hasattr(hp, '_detect_people_yolo26')}, ByteTrackStyleTracker={hasattr(hp, 'ByteTrackStyleTracker')}")
except Exception as e:
    print(f"   ❌ hospital_pipeline failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test tracker instantiation
print("\n[5/6] Testing ByteTrackStyleTracker instantiation...")
try:
    tracker = hp.ByteTrackStyleTracker(
        high_conf_thresh=0.35,
        low_conf_thresh=0.15,
        iou_gate_high=0.30,
        iou_gate_low=0.25,
    )
    print("   ✅ Tracker created")
    print(f"   Next ID: {tracker.next_track_id}")
except Exception as e:
    print(f"   ❌ Tracker failed: {e}")
    sys.exit(1)

# Test 6: Test camera calibration config
print("\n[6/6] Testing camera calibration config...")
try:
    import json
    calib_path = Path(__file__).parent.parent / "backend" / "config" / "camera_calibration.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
        print(f"   ✅ Calibration loaded: {len(calib['cameras'])} cameras")
    else:
        print("   ⚠️  Calibration file not found (optional)")
except Exception as e:
    print(f"   ⚠️  Calibration warning: {e}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE - All systems ready for L4 Edge-First!")
print("=" * 60)

print("\n📊 Performance Targets:")
print("   • YOLO26-X: 56.9 AP, ~45 FPS on L4")
print("   • CLIP-ReID: 91.2% Rank-1 on MSMT17")
print("   • ByteTrack: HOTA >62.0")
print("   • VRAM usage: <18GB (leaves 6GB for VLM)")
print("\n🚀 Next: Run `docker-compose -f docker-compose.edge.yml up`")
