import pytest
import torch

from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel


def test_repair_changes_require_explicit_variant():
    VLDriveTrainingConfig().validate()
    for changed in ({'reconstruction_reduction': 'mean'}, {'detach_reconstruction_loss_mask': True},
                    {'reconstruction_feature_space': 'normalized'},
                    {'inference_selection_mode': 'topk'}, {'retention_logit_weight': 1.}):
        with pytest.raises(ValueError, match='explicit'):
            VLDriveTrainingConfig(**changed).validate()
        VLDriveTrainingConfig(training_variant='morai_repair_v1', **changed).validate()


def make_output():
    trajectory = torch.ones((1, 3, 5, 2), requires_grad=True)
    return dict(trajectory=trajectory, cycle_loss=torch.tensor(0.), pruning_loss=torch.tensor(0.),
                reconstruction_loss=torch.tensor(0.))


def test_instruction_boundary_loss_does_not_supervise_old_instruction_frames():
    output = make_output()
    targets = dict(future_trajectory=torch.zeros_like(output['trajectory']),
                   frame_loss_mask=torch.tensor([[False, True, True]]))
    loss = VLDriveTrainingModel.loss(output, targets)
    assert loss['waypoint_loss'].item() == 1.
    loss['loss'].backward()
    assert output['trajectory'].grad[0, 0].abs().sum() == 0
    assert output['trajectory'].grad[0, 1:].abs().sum() > 0


@pytest.mark.parametrize('mask', [torch.tensor([[True, False, True]]), torch.tensor([[True, True, False]]),
                                  torch.tensor([[False, False, False]]), torch.ones((1, 3))])
def test_bad_instruction_loss_mask_rejected(mask):
    output = make_output()
    with pytest.raises(ValueError, match='suffix'):
        VLDriveTrainingModel.loss(output, dict(future_trajectory=torch.zeros_like(output['trajectory']),
                                               frame_loss_mask=mask))


def test_legacy_full_frame_l1_is_unchanged():
    output = make_output()
    target = torch.zeros_like(output['trajectory'])
    assert VLDriveTrainingModel.loss(output, {'future_trajectory': target})['waypoint_loss'] == 1.
