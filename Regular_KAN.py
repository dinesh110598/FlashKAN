# %%
import torch
from torch import nn
from torch.func import vmap, grad
from splines import _evaluate, evaluate_diff
# %%
class Regular_KAN(nn.Module):
    def __init__(self, in_dim, out_dim, G, 
                 t0=0., t1=1., k=4, act=nn.functional.silu):
        super().__init__()
        w = torch.empty([(G+k)*in_dim, out_dim])
        self.w = nn.Parameter(torch.nn.init.xavier_normal_(w), 
                              requires_grad=True)
        # B spline order and knots
        self.k = k
        t = torch.linspace(t0, t1, G+1)
        self.t = nn.Parameter(torch.cat([torch.full([k-1], t0), t,
                              torch.full([k-1], t1)], 0), 
                              requires_grad=False)
        self.act = act
        
    def forward(self, x):
        w, t, k, act = self.w, self.t, self.k, self.act
        batch_dim = x.shape[0]
        
        y1 = bspline_evaluate.apply(x, t, k) #[batch, in, G+k-1]
        y2 = torch.cat([y1, act(x).unsqueeze(2)], dim=2)
        return torch.matmul(y2.reshape(batch_dim, -1), w)
                

class bspline_evaluate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t, k):
        i_len = t.shape[0]-k
        i_full = torch.arange(i_len).reshape([1,1,i_len]).int()
        i_full = torch.broadcast_to(i_full, [*x.shape]+[i_len])
        
        ctx.save_for_backward(x, t, i_full)
        ctx.k = k
        return vmap(_evaluate, (None, None, 2, None), 2)(k, t, i_full, x)
        
    @staticmethod
    def backward(ctx, d_out):
        x, t, i_full = ctx.saved_tensors
        k = ctx.k
        
        Dx = (d_out*vmap(evaluate_diff, (None, None, 2, None), 2)(
            k, t, x, i_full
        )).sum(2)
        
        return Dx, None, None
        