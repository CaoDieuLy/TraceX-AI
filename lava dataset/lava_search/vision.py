from __future__ import annotations

from pathlib import Path


def detect_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_rows(array):
    import numpy as np

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def encode_texts_sentence_transformer(
    texts: list[str],
    model_name: str,
    device: str | None = None,
    batch_size: int = 16,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense text embeddings."
        ) from exc

    model = SentenceTransformer(model_name, device=detect_device(device))
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.astype("float32")


def _load_open_clip(model_name: str, pretrained: str, device: str | None = None):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open-clip-torch is required for multimodal CLIP indexing."
        ) from exc

    resolved_device = detect_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=resolved_device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return {
        "device": resolved_device,
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
    }


def encode_texts_clip(
    texts: list[str],
    model_name: str,
    pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
):
    import numpy as np

    bundle = _load_open_clip(model_name, pretrained, device=device)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    resolved_device = bundle["device"]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CLIP text encoding.") from exc

    outputs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = tokenizer(batch).to(resolved_device)
            embeddings = model.encode_text(tokens).detach().cpu().numpy()
            outputs.append(embeddings)
    return _normalize_rows(np.vstack(outputs)).astype("float32")


def encode_images_clip(
    image_paths: list[str | Path],
    model_name: str,
    pretrained: str,
    device: str | None = None,
    batch_size: int = 16,
):
    import numpy as np

    bundle = _load_open_clip(model_name, pretrained, device=device)
    model = bundle["model"]
    preprocess = bundle["preprocess"]
    resolved_device = bundle["device"]

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CLIP image encoding.") from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for CLIP image encoding.") from exc

    tensors = []
    outputs = []
    with torch.no_grad():
        for image_path in image_paths:
            tensor = preprocess(Image.open(image_path).convert("RGB"))
            tensors.append(tensor)
            if len(tensors) >= batch_size:
                batch = torch.stack(tensors).to(resolved_device)
                embeddings = model.encode_image(batch).detach().cpu().numpy()
                outputs.append(embeddings)
                tensors = []

        if tensors:
            batch = torch.stack(tensors).to(resolved_device)
            embeddings = model.encode_image(batch).detach().cpu().numpy()
            outputs.append(embeddings)

    return _normalize_rows(np.vstack(outputs)).astype("float32")
