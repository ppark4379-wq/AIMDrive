"""Independent scalar and gradient checks for the fixed N*C debug reduction."""
from dataclasses import replace
import pytest
import torch
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig


@pytest.mark.parametrize('kept', [1,2,3,4])
def test_fixed_nc_uses_capacity_not_drop_count_and_preserves_mask_gradient(kept):
    torch.manual_seed(603)
    config = QFormerMEFAConfig(visual_dim=8,num_visual_tokens=4,language_dim=16,
        query_count=2,memory_size=2,qformer_hidden_size=16,qformer_num_layers=1,
        qformer_num_attention_heads=2,qformer_intermediate_size=32,
        reconstruction_layers=1,reconstruction_heads=2)
    connector = QFormerMEFAConnector(config).eval()
    class ZeroReconstruction(torch.nn.Module):
        def forward(self, value): return value*0
    connector.reconstruction_transformer = ZeroReconstruction()
    features = torch.arange(64,dtype=torch.float32).reshape(1,2,4,8)/8+3
    mask = torch.zeros(1,2,4)
    mask[:,:,:kept] = 1
    mask.requires_grad_()
    outputs, grads = [], []
    for reduction in ('sum','fixed_nc'):
        connector.config = replace(config, reconstruction_reduction=reduction)
        out = connector(features, hard_mask=mask, compute_reconstruction=True)
        outputs.append(out)
        grads.append(torch.autograd.grad(out['reconstruction_loss'],mask)[0])
    expected_sum = sum(float(features[0,t,j,c])**2 * (1-float(mask[0,t,j].detach()))
        for t in range(2) for j in range(4) for c in range(8))/2
    torch.testing.assert_close(outputs[0]['reconstruction_loss'],torch.tensor(expected_sum))
    torch.testing.assert_close(outputs[1]['reconstruction_loss'],torch.tensor(expected_sum/(4*8)))
    torch.testing.assert_close(grads[1],grads[0]/32,rtol=1e-6,atol=1e-6)
    assert grads[1].abs().sum() > 0  # Even all-KEEP still receives the outer-mask derivative.
    for name in ('language_tokens','selection_mask','reconstructed_features','reconstruction_target','pruning_loss'):
        torch.testing.assert_close(outputs[0][name],outputs[1][name],rtol=0,atol=0)


def test_fixed_nc_debug_is_explicit_and_cannot_enable_other_repairs():
    assert VLDriveTrainingConfig().reconstruction_reduction == 'sum'
    base = VLDriveTrainingConfig(training_variant='fixed_nc_debug',reconstruction_reduction='fixed_nc')
    base.validate()
    for changes in ({'training_variant':'paper_independent'}, {'reconstruction_reduction':'sum'},
                    {'detach_reconstruction_loss_mask':True}, {'inference_selection_mode':'topk'},
                    {'reconstruction_feature_space':'normalized'}, {'retention_logit_weight':1.},
                    {'selector_task_gradient_mode':'retained_output_st'}, {'pruning_weight':1.}):
        with pytest.raises(ValueError):
            replace(base,**changes).validate()
