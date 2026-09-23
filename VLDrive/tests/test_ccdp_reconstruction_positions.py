"""Identifiable-token oracle for retained slot order and Eq.(8) scatter."""
import torch
from models.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector


def test_distinct_tokens_survive_gather_mefa_padding_scatter_in_original_positions():
    cfg=QFormerMEFAConfig(visual_dim=8,num_visual_tokens=6,language_dim=8,
        query_count=2,memory_size=2,qformer_hidden_size=8,qformer_num_layers=1,
        qformer_num_attention_heads=2,qformer_intermediate_size=16,
        reconstruction_heads=2)
    net=QFormerMEFAConnector(cfg).eval()
    # Test doubles expose identity labels across the real packing/scattering
    # flow; these replacements are confined to this synthetic unit fixture.
    for name in ('visual_input_norm','retained_query_projection','qformer_to_language',
                 'reconstruction_align','reconstruction_transformer'):
        setattr(net,name,torch.nn.Identity())
    features=torch.arange(2*3*6*8,dtype=torch.float32).reshape(2,3,6,8)+100
    masks=torch.tensor([[[1,0,1,0,0,1],[0,1,0,0,1,0],[0,0,0,1,0,0]],
                        [[0,1,1,0,1,0],[1,0,0,0,0,0],[1,1,1,1,1,1]]],dtype=torch.float32)
    raw=features.flatten(0,1); flatmask=masks.flatten(0,1)
    def identifiable_mefa(query_embeddings,query_valid_mask,memory,**kwargs):
        assert memory.shape[:2]==(6,6)  # Full N K/V tokens are preserved.
        for f in range(6):
            expected=[j for j in range(6) if flatmask[f,j]==1]
            assert int(query_valid_mask[f].sum())==len(expected)
            for slot,j in enumerate(expected):
                torch.testing.assert_close(query_embeddings[f,slot],raw[f,j],rtol=0,atol=0)
        # Deliberately contaminate padded slots: the actual valid-mask path
        # must prevent them from being scattered into source index zero.
        return torch.where(query_valid_mask[...,None],query_embeddings+10000,
                           torch.full_like(query_embeddings,999999)),[]
    net._run_qformer=identifiable_mefa
    with torch.no_grad():
        net.mask_embedding.copy_(-torch.arange(48).reshape(1,6,8)-10)
        net.reconstruction_position_embedding.copy_(torch.arange(48).reshape(1,6,8)/100)
    out=net(features,hard_mask=masks,compute_reconstruction=True)
    base=out['reconstruction_base_input']
    for f in range(6):
        expected_indices=[j for j in range(6) if flatmask[f,j]==1]
        got=out['selected_indices'].reshape(6,-1)[f]
        assert got[:len(expected_indices)].tolist()==expected_indices
        assert (got[len(expected_indices):]==-1).all()
        for j in range(6):
            expected=raw[f,j]+10000 if flatmask[f,j]==1 else net.mask_embedding[0,j]
            torch.testing.assert_close(base[f,j],expected,rtol=0,atol=0)
    torch.testing.assert_close(out['reconstruction_input'],base+net.reconstruction_position_embedding,rtol=0,atol=0)
    torch.testing.assert_close(out['reconstruction_target'],raw,rtol=0,atol=0)
    assert net.mask_embedding.shape==(1,6,8)
    assert net.mask_embedding.requires_grad


def test_postnorm_transformer_exposes_learnable_output_affine_without_extra_final_norm():
    cfg=QFormerMEFAConfig(visual_dim=8,num_visual_tokens=6,language_dim=8,
        query_count=2,memory_size=2,qformer_hidden_size=8,qformer_num_layers=1,
        qformer_num_attention_heads=2,qformer_intermediate_size=16,reconstruction_heads=2)
    net=QFormerMEFAConnector(cfg)
    layers=net.reconstruction_transformer.layers
    assert len(layers)==4 and all(not x.norm_first for x in layers)
    assert net.reconstruction_transformer.norm is None
    assert all(x.norm2.elementwise_affine and x.norm2.weight.requires_grad for x in layers)
