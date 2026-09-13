"""Run a real MORAI batch through Memfuser, training losses and backward.

Uses random weights; no checkpoint is downloaded or written. This validates
the computation contract, not driving quality or checkpoint compatibility.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from timm.data import create_morai_dataset, create_morai_loader
from timm.models.memfuser import Memfuser
from timm.loss import LabelSmoothingCrossEntropy
from train_pretrain import LAVLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--index-file", default="dataset_index.txt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    dataset = create_morai_dataset(
        "morai", root=args.dataset_root, index_filename=args.index_file,
        with_lidar=True, multi_view=True,
    )
    loader = create_morai_loader(
        dataset, input_size=(3, 224, 224), multi_view_input_size=(3, 128, 128),
        batch_size=args.batch_size, num_workers=0, is_training=False,
    )
    inputs, targets = next(iter(loader))
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    targets = [value.to(args.device) for value in targets]
    # Same dimensions and heads as memfuser_baseline_e1d3, without weight download.
    model = Memfuser(
        enc_depth=1, dec_depth=3, embed_dim=256, rgb_backbone_name="r50",
        lidar_backbone_name="conv", waypoints_pred_head="gru",
        pretrained_backbone=False,
    ).to(args.device).train()
    outputs = model(inputs)
    batch = inputs["rgb_front"].shape[0]
    expected = [(batch, 2500, 8), (batch, 5, 2), (batch, 2), (batch, 2)]
    for output, shape in zip(outputs[:4], expected):
        assert tuple(output.shape) == shape, (output.shape, shape)
        assert torch.isfinite(output).all()
    traffic, velocity = LAVLoss()(outputs[0], targets[4])
    classification = LabelSmoothingCrossEntropy(smoothing=0.1)
    terms = {
        "traffic": traffic * 0.5,
        "waypoints": torch.nn.functional.l1_loss(outputs[1], targets[1]) * 0.5,
        "velocity": velocity * 0.05,
        "traffic_light": classification(outputs[2], targets[3]) * 0.1,
        "stop_sign": classification(outputs[3], targets[6]) * 0.01,
    }
    loss = sum(terms.values())
    assert torch.isfinite(loss), terms
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for name, component in (
        ("camera", model.rgb_backbone), ("lidar", model.lidar_backbone),
        ("waypoints", model.waypoints_generator),
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in component.parameters()), name
    torch.optim.SGD(model.parameters(), lr=1e-5).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print("outputs:", [tuple(output.shape) for output in outputs])
    print("weighted losses:", {name: float(value.detach()) for name, value in terms.items()})
    print("PASS: forward, all training losses, finite gradients and optimizer step")


if __name__ == "__main__":
    main()
