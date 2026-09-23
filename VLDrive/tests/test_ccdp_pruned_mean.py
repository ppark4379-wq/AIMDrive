"""Independent quotient-rule oracle, including frame scope and zero drops."""
from dataclasses import replace
import pytest
import torch
from models.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig


def make_connector():
    cfg=QFormerMEFAConfig(visual_dim=8,num_visual_tokens=4,language_dim=16,
        query_count=2,memory_size=2,qformer_hidden_size=16,qformer_num_layers=1,
        qformer_num_attention_heads=2,qformer_intermediate_size=32,
        reconstruction_layers=1,reconstruction_heads=2,reconstruction_reduction='pruned_mean')
    net=QFormerMEFAConnector(cfg).eval()
    class Zero(torch.nn.Module):
        def forward(self,x): return x*0
    net.reconstruction_transformer=Zero()
    return net


def test_pruned_mean_quotient_gradient_and_per_frame_scope():
    net=make_connector()
    x=torch.arange(64,dtype=torch.float32).reshape(1,2,4,8)/8+1
    m=torch.tensor([[[1.,0.,0.,0.],[1.,1.,1.,0.]]],requires_grad=True)
    out=net(x,hard_mask=m,compute_reconstruction=True)
    per_token=x.square().mean(-1)
    expected=[]; grad=torch.empty_like(m)
    for t in range(2):
        dropped=[j for j in range(4) if float(m[0,t,j].detach())==0]
        avg=sum(float(per_token[0,t,j]) for j in dropped)/len(dropped)
        expected.append(avg)
        for j in range(4):
            grad[0,t,j]=(avg-float(per_token[0,t,j]))/len(dropped)/2
    torch.testing.assert_close(out['reconstruction_loss'],torch.tensor(sum(expected)/2))
    actual=torch.autograd.grad(out['reconstruction_loss'],m)[0]
    torch.testing.assert_close(actual,grad,rtol=1e-5,atol=1e-5)
    assert (actual>0).any() and (actual<0).any()
    global_mean=((1-m)*per_token).sum()/(1-m).sum()
    assert not torch.isclose(out['reconstruction_loss'],global_mean)


def test_zero_drops_is_finite_zero_without_detaching_mask():
    net=make_connector()
    x=torch.randn(1,2,4,8)
    m=torch.ones(1,2,4,requires_grad=True)
    loss=net(x,hard_mask=m,compute_reconstruction=True)['reconstruction_loss']
    grad=torch.autograd.grad(loss,m)[0]
    assert loss.item()==0 and torch.isfinite(grad).all() and grad.abs().sum()>0


def test_pruned_mean_requires_isolated_opt_in():
    cfg=VLDriveTrainingConfig(training_variant='pruned_mean_debug',reconstruction_reduction='pruned_mean')
    cfg.validate()
    for changes in ({'training_variant':'paper_independent'}, {'reconstruction_reduction':'sum'},
        {'detach_reconstruction_loss_mask':True},{'inference_selection_mode':'topk'},
        {'reconstruction_feature_space':'normalized'},{'retention_logit_weight':1.}):
        with pytest.raises(ValueError): replace(cfg,**changes).validate()
    assert VLDriveTrainingConfig().reconstruction_reduction=='sum'
