import torch

from utils.metrics import TrajectoryMetrics


def test_axis_mae_agrees_between_stage6e_and_lmdrive_coordinates():
    # The same physical error: four metres forward and three metres left.
    # LMDrive stores left in x and negative forward in y.
    reports = []
    for kwargs, xy in (
        ({}, [4.0, 3.0]),
        ({"coordinate_convention": "lmdrive_left_negative_forward"}, [3.0, -4.0]),
    ):
        metrics = TrajectoryMetrics(**kwargs)
        prediction = torch.tensor(xy).reshape(1, 1, 2).repeat(1, 5, 1)
        metrics.update(prediction, torch.zeros_like(prediction), ["route"], ["route/0010"])
        report = metrics.compute()
        assert report["longitudinal_mae_m"] == [4.0] * 5
        assert report["lateral_mae_m"] == [3.0] * 5
        assert report["per_run"]["route"]["longitudinal_mae_m"] == [4.0] * 5
        assert report["per_run"]["route"]["lateral_mae_m"] == [3.0] * 5
        assert report["ade_m"] == 5.0
        assert report["fde_m"] == 5.0
        reports.append(report)
    assert reports[0]["coordinate_convention"] == "forward_left"
    assert reports[1]["coordinate_convention"] == "lmdrive_left_negative_forward"
