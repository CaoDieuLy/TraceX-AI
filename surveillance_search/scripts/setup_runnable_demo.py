from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a runnable local demo by downloading public data, prewarming models, and optionally building the bundle."
    )
    parser.add_argument(
        "--dataset-type",
        choices=("personpath22",),
        default="personpath22",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--locations", default=None)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--prewarm-models", action="store_true")
    parser.add_argument("--register-baselines", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--assets-dir", default=None)
    return parser


def _write_scaffold_manifests(project_root: Path) -> None:
    from surveillance_search.person_query_stack import ensure_person_query_stack_layout, write_stack_configs

    layout = ensure_person_query_stack_layout(project_root)
    write_stack_configs(project_root)

    pa_manifest = layout["pa100k_prepared_root"] / "manifest.json"
    if not pa_manifest.exists():
        pa_manifest.write_text(
            json.dumps(
                {
                    "dataset": "PA-100K",
                    "raw_root": str(layout["pa100k_root"]),
                    "images_dir": str(layout["pa100k_root"] / "images"),
                    "annotation_file": None,
                    "status": "prepared_scaffold",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    bdd_manifest = layout["bdd100k_prepared_root"] / "manifest.json"
    if not bdd_manifest.exists():
        bdd_manifest.write_text(
            json.dumps(
                {
                    "dataset": "BDD100K",
                    "raw_root": str(layout["bdd100k_root"]),
                    "images_dir": str(layout["bdd100k_root"] / "images"),
                    "labels_dir": str(layout["bdd100k_root"] / "labels"),
                    "status": "prepared_scaffold",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _register_baselines(project_root: Path) -> None:
    import subprocess

    commands = [
        [sys.executable, str(project_root / "scripts" / "train_attribute_baseline.py")],
        [sys.executable, str(project_root / "scripts" / "train_scene_baseline.py")],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=project_root)


def _prewarm_models(args: argparse.Namespace) -> None:
    from surveillance_search.vision import encode_texts_clip, encode_texts_sentence_transformer

    encode_texts_sentence_transformer(
        ["surveillance retrieval warmup"],
        model_name=args.sentence_model,
        device=args.device,
        batch_size=1,
    )
    encode_texts_clip(
        ["surveillance retrieval warmup"],
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=1,
    )


def main() -> int:
    args = build_parser().parse_args()
    project_root = _project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from surveillance_search.config import default_bundle_root, default_data_root, default_visual_root
    from surveillance_search.dataset import download_from_huggingface
    from surveillance_search.runtime import RuntimeConfig, rebuild_runtime_bundle

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else default_data_root(args.dataset_type)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_bundle_root()
    assets_dir = Path(args.assets_dir).expanduser().resolve() if args.assets_dir else default_visual_root()

    if args.register_baselines:
        _write_scaffold_manifests(project_root)
        _register_baselines(project_root)

    if args.download_data:
        download_from_huggingface(
            dataset_root=dataset_root,
            repo_id="",
            locations=_parse_csv(args.locations),
            splits=_parse_csv(args.splits),
            include_videos=args.include_videos,
            include_docs=False,
            dataset_type=args.dataset_type,
        )

    if args.prewarm_models:
        _prewarm_models(args)

    if args.build:
        config = RuntimeConfig(
            dataset_type=args.dataset_type,
            dataset_root=dataset_root,
            output_dir=output_dir,
            assets_dir=assets_dir,
            locations=_parse_csv(args.locations) or None,
            splits=_parse_csv(args.splits) or None,
            group_by_track=True,
            fps=30.0,
            enable_sparse=True,
            enable_dense=True,
            enable_clip=True,
            sentence_model_name=args.sentence_model,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            device=args.device,
            batch_size=args.batch_size,
            max_assets_per_moment=3,
            crop_padding=0.08,
            ffmpeg_bin="ffmpeg",
            enable_enrichment=False,
        )
        manifest = rebuild_runtime_bundle(config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("Setup completed.")
    print(f"Dataset root: {dataset_root}")
    print(f"Bundle root:  {output_dir}")
    print(f"Assets root:  {assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
