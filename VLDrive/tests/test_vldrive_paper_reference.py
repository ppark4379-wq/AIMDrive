"""Independent scalar reference for the joint denominators in paper Eq. (9)."""
import math

import torch

from models.llm.ddia_attention import apply_rope, compose_ddia_logits
from models.llm.multimodal_sequence import INSTRUCTION, VISUAL, PADDING


def test_ddia_matches_scalar_paper_equation_with_one_joint_denominator():
    torch.manual_seed(418)
    q=torch.randn(1,2,6,4,dtype=torch.float64)
    k=torch.randn_like(q)
    v=torch.randn_like(q)
    positions=torch.tensor([[0,1,17,31,1250,1251]])
    modes=torch.tensor([[INSTRUCTION,INSTRUCTION,VISUAL,VISUAL,VISUAL,PADDING]])
    qr=apply_rope(q,positions)
    kr=apply_rope(k,positions)
    logits,allowed=compose_ddia_logits(q,k,qr,kr,modes)
    actual=torch.softmax(logits[:,:,:5].masked_fill(~allowed[:,:,:5],-torch.inf),-1) @ v
    expected=torch.zeros_like(actual)
    for h in range(2):
        for j in range(5):
            keys=list(range(2)) if j<2 else list(range(j))
            scores=[]
            for i in keys:
                cross=j>=2 and i<2
                a,b=(q[0,h,j],k[0,h,i]) if cross else (qr[0,h,j],kr[0,h,i])
                scores.append(math.exp(sum(float(a[d])*float(b[d]) for d in range(4))/2))
            denominator=sum(scores)
            for i,score in zip(keys,scores):
                expected[0,h,j]+=v[0,h,i]*(score/denominator)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    shifted=positions.clone()
    shifted[:,2:]+=400
    changed,_=compose_ddia_logits(q,k,apply_rope(q,shifted),apply_rope(k,shifted),modes)
    torch.testing.assert_close(changed[:,:,2:5,:2],logits[:,:,2:5,:2],rtol=0,atol=0)


def test_equation13_reconstructs_frozen_encoder_features_not_trainable_normalization():
    from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
    config=QFormerMEFAConfig(visual_dim=8,num_visual_tokens=4,language_dim=16,
        query_count=2,memory_size=2,qformer_hidden_size=16,qformer_num_layers=1,
        qformer_num_attention_heads=2,qformer_intermediate_size=32,
        reconstruction_layers=1,reconstruction_heads=2)
    connector=QFormerMEFAConnector(config).eval()
    class ZeroReconstruction(torch.nn.Module):
        def forward(self, value): return value*0
    connector.reconstruction_transformer=ZeroReconstruction()
    features=torch.arange(64,dtype=torch.float32).reshape(1,2,4,8)/8+3
    mask=torch.tensor([[[1.,0.,1.,0.],[0.,1.,0.,1.]]])
    expected=sum(float(features[0,t,j,c])**2
        for t in range(2) for j in range(4) for c in range(8) if mask[0,t,j]==0)/2
    for weight,bias in ((1.,0.),(2.,17.)):
        with torch.no_grad():
            connector.visual_input_norm.weight.fill_(weight)
            connector.visual_input_norm.bias.fill_(bias)
        output=connector(features,hard_mask=mask,compute_reconstruction=True)
        torch.testing.assert_close(output['reconstruction_target'],features.flatten(0,1),rtol=0,atol=0)
        assert not output['reconstruction_target'].requires_grad
        torch.testing.assert_close(output['reconstruction_loss'],torch.tensor(expected))
    # The full reconstruction remains a training-only auxiliary by default.
    assert not connector(features,hard_mask=mask)['reconstruction_computed']
