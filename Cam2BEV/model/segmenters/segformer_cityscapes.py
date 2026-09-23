"""Pretrained RGB -> Cam2BEV IDs. CPU default; no training performed."""
import argparse
import os
from pathlib import Path
import numpy as np

MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
# Cityscapes train IDs -> Cam2BEV input channels (not XML toValue numbers).
# rider/motorcycle/bicycle -> bike; train -> obstacle; terrain -> vegetation.
CITYSCAPES_TO_CAM2BEV = np.array(
    [0, 1, 7, 7, 7, 7, 7, 7, 8, 8, 9, 2, 6, 3, 4, 5, 7, 6, 6], dtype=np.uint8)
EXPECTED_LABELS = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle")
PALETTE = np.array([
    [128,64,128], [244,35,232], [255,0,0], [0,0,142], [0,0,70],
    [0,60,100], [220,20,60], [0,0,0], [107,142,35], [70,130,180]], dtype=np.uint8)
_runtime = None


def map_classes(ids):
    ids = np.asarray(ids)
    if not np.issubdtype(ids.dtype, np.integer) or ids.size == 0:
        raise ValueError("Cityscapes IDs must be a nonempty integer array")
    if ids.min() < 0 or ids.max() >= len(CITYSCAPES_TO_CAM2BEV):
        raise ValueError("Cityscapes ID is outside 0..18")
    return CITYSCAPES_TO_CAM2BEV[ids]


def _load():
    global _runtime
    if _runtime is None:
        # Transformers uses PyTorch here; Cam2BEV itself can still use TensorFlow.
        os.environ.setdefault("USE_TF", "0")
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        device = os.environ.get("CAM2BEV_SEGMENTER_DEVICE", "cpu")
        model_path = os.environ.get("CAM2BEV_SEGMENTER_MODEL", MODEL_ID)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        labels = tuple(model.config.id2label[i] for i in range(model.config.num_labels))
        if labels != EXPECTED_LABELS:
            raise ValueError("This adapter requires the Cityscapes 19-class label order")
        model.to(device).eval()
        _runtime = (torch, processor, model, device)
    return _runtime


def segment(rgb_image, camera_name=None):
    """Return native HxW Cam2BEV integer IDs for an RGB uint8 image."""
    rgb_image = np.asarray(rgb_image)
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != np.uint8:
        raise ValueError("Expected HxWx3 RGB uint8 image")
    torch, processor, model, device = _load()
    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs)
        # Resize logits to original camera dimensions BEFORE argmax.
        ids = processor.post_process_semantic_segmentation(
            output, target_sizes=[rgb_image.shape[:2]])[0].cpu().numpy()
    return map_classes(ids)


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description="Inspect pretrained segmentation on a saved MORAI RGB frame")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for segmentation outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    ids = segment(rgb)
    colors = PALETTE[ids]
    Image.fromarray(ids).save(args.output / "class_ids.png")
    Image.fromarray(colors).save(args.output / "semantic.png")
    Image.fromarray(((rgb.astype(np.float32) + colors) / 2).astype(np.uint8)).save(args.output / "overlay.png")
    Image.fromarray((ids == 0).astype(np.uint8) * 255).save(args.output / "road_mask.png")
    print("Saved camera-view segmentation (not BEV) to", args.output)


if __name__ == "__main__":
    main()
