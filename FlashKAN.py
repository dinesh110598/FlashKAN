# %%
import torch
from torch import nn
from torch.func import vmap, grad
from splines import evaluate_all, evaluate_diff_idx
# %%
class FlashKAN(nn.Module):
    def __init__(self, in_dim, out_dim, G, 
                 t0=-1., t1=1., k=4, act=nn.functional.silu):
        super().__init__()
        w = torch.empty([G+k, in_dim, out_dim])
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
        return fast_kan.apply(x, w, t, k, act)
# %% Custom gradients
class fast_kan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, t, k, act):
        in_dim, out_dim = x.shape[1], w.shape[2]
        ctx.k = k
        ctx.act = act
        
        i_full, y1 = evaluate_all(k, t, x)
        # k-tuples of arrays with shape (batch, in_dim)
        
        i_full += [torch.full_like(i_full[0], -1)]
        # Add the last 'row' of w
        # Slice w to shape [batch, in, k+1, out]
        slice1 = torch.stack(i_full, 2).unsqueeze(3)
        slice2 = torch.reshape(torch.arange(in_dim, device=slice1.device), 
                               [1,in_dim,1,1])
        slice3 = torch.reshape(torch.arange(out_dim, device=slice1.device), 
                               [1,1,1,out_dim])

        w2 = w[slice1, slice2, slice3] # [batch, in, k+1, out]
        
        y1 += [act(x)]
        y2 = torch.stack(y1, 2).unsqueeze(3) # [batch, in, k+1, 1]
        
        ctx.save_for_backward(x, w, t, slice1, y2)
        
        return torch.sum(w2 * y2, dim=(1,2))
        
    @staticmethod
    def backward(ctx, D_out):
        x, w, t, slice1, y2 = ctx.saved_tensors
        k, act = ctx.k, ctx.act
        in_dim, out_dim, batch_dim = x.shape[1], w.shape[2], x.shape[0]
        
        Dw = torch.zeros_like(w)
        slice2 = torch.reshape(torch.arange(in_dim, device=slice1.device), 
                               [1,in_dim,1,1])
        slice3 = torch.reshape(torch.arange(out_dim, device=slice1.device), 
                               [1,1,1,out_dim])
        D_out = torch.reshape(D_out, [batch_dim, 1, 1, out_dim])
        Dw[slice1, slice2, slice3] += (y2 * D_out)
        
        # Use F.grad with vmap to compute per sample gradients
        Dy1 = evaluate_diff_idx(k, t, x, slice1[:, :, :-1, 0].unbind(2))
        Dy1 += [vmap(grad(act))(x.reshape(-1)).reshape_as(x)]
        Dy2 = torch.stack(Dy1, 2).unsqueeze(3) # [batch, in, k+1, 1]
        
        w2 = w[slice1, slice2, slice3] # [batch, in, k+1, out]
        Dx = torch.sum(D_out * w2 * Dy2, dim=(2,3))
        
        return Dx, Dw, None, None, None
# %%