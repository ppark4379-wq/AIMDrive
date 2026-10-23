import unittest
import torch
from torch import nn
from models.ve.transformer import TransformerDecoderLayer
from training.vision_encoder.losses.morai_geometry_decoder import GeometryDecoderBranch


class GeometryDecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.layer=TransformerDecoderLayer(16,4,32,.1,nn.GELU).eval()
        self.norm=nn.LayerNorm(16).eval()
        self.head=nn.Sequential(nn.Linear(48,16),nn.ReLU(),nn.Linear(16,8)).eval()
        self.tgt=torch.randn(9,2,16);self.mem=torch.randn(5,2,16);self.pos=torch.randn(9,2,16)
        self.vel=torch.randn(2,9,32)
        captured=[]
        h=self.layer.norm1.register_forward_hook(lambda m,a,o:captured.append(o.detach()))
        with torch.no_grad():
            out=self.norm(self.layer(self.tgt,self.mem,query_pos=self.pos)).transpose(0,1)
            self.base=self.head(torch.cat((out,self.vel),-1))
        h.remove();self.res=captured[0].transpose(0,1);self.query=(captured[0]+self.pos).transpose(0,1)

    def branch(self, mode='fusion'):
        return GeometryDecoderBranch(self.layer,self.norm,self.head,mode)

    def test_initial_branch_reproduces_original_for_selected_queries(self):
        ix=[1,4,7];b=self.branch()
        p=b(self.res[:,ix],self.query[:,ix],self.mem.transpose(0,1),self.vel[:,ix],self.base[:,ix])
        torch.testing.assert_close(p,self.base[:,ix],atol=1e-6,rtol=0)

    def test_training_only_changes_copy_and_geometry_output(self):
        before={k:v.clone() for k,v in self.layer.state_dict().items()}
        b=self.branch();b.train();opt=torch.optim.AdamW(b.parameters(),lr=.01)
        self.assertFalse(b.multihead_attn.training)
        p=b(self.res,self.query,self.mem.transpose(0,1),self.vel,self.base)
        opt.zero_grad();p[...,1:7].square().mean().backward();opt.step()
        q=b(self.res,self.query,self.mem.transpose(0,1),self.vel,self.base)
        self.assertTrue(torch.equal(q[...,[0,7]],self.base[...,[0,7]]))
        self.assertFalse(torch.equal(q[...,1:7],p[...,1:7]))
        for k,v in self.layer.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertIsNotNone(b.multihead_attn.in_proj_weight.grad)

    def test_head_only_control_has_no_fusion_gradients(self):
        b=self.branch('head_only')
        p=b(self.res,self.query,self.mem.transpose(0,1),self.vel,self.base)
        p[...,1:7].square().mean().backward()
        self.assertIsNone(b.multihead_attn.in_proj_weight.grad)
        self.assertIsNotNone(b.head[0].weight.grad)


if __name__=='__main__':unittest.main()
