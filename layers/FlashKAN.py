# %%
import torch
from torch import nn
from torch.func import vmap, grad
from .splines import evaluate_all, b_spline_diff
# %%
class FlashKAN(nn.Module):
    def __init__(self, in_dim, out_dim, G, 
                 t0=-1., t1=1., k=4, act=nn.functional.silu):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.G = G
        self.k = k
        
        w = torch.empty([G+k, in_dim, out_dim])
        self.w = nn.Parameter(torch.nn.init.xavier_normal_(w), 
                              requires_grad=True)
        # B spline order and knots
        self.k = k
        t = torch.linspace(t0, t1, G+1)
        t = torch.cat([torch.full([k-1], t0), t,
                        torch.full([k-1], t1)], 0)
        self.register_buffer("t", t)
        self.act = act
        
    def forward(self, x):
        return fast_kan.apply(x, self.w, self.t, 
                              self.k, self.act)
    
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.k):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).
            y (torch.Tensor): Output tensor of shape (batch_size, out_dim).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_dim, in_dim, G+k)
        """
        assert x.dim() == 2 and x.size(1) == self.in_dim
        assert y.size() == (x.size(0), self.out_dim)

        A = self.b_splines(x)  # (batch_size, in_dim, G + k)
        solution = torch.linalg.lstsq(
            A.flatten(1,-1), y
        ).solution  # (in_dim*(G+k), out_dim)
        result = solution.reshape(A.shape[2], A.shape[1], -1)
        
        return result.contiguous()

# %% Custom gradients
class fast_kan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, t, k, act):
        in_dim, out_dim = x.shape[1], w.shape[2]
        ctx.k = k
        ctx.act = act
        
        i_full, y1 = evaluate_all(k, t, x)
        # (batch, in_dim, 2k), (batch, in_dim, k)
        
        # Add the last 'row' of w
        slice1 = torch.cat([i_full[..., :k], 
                            torch.full_like(i_full[..., :1], -1)], 
                           -1).unsqueeze(-1)
        # Slice w to shape [batch, in, k+1, out]
        slice2 = torch.reshape(torch.arange(in_dim, device=slice1.device), 
                               [1,in_dim,1,1])
        slice3 = torch.reshape(torch.arange(out_dim, device=slice1.device), 
                               [1,1,1,out_dim])

        w2 = w[slice1, slice2, slice3] # [batch, in, k+1, out]
        
        y2 = torch.cat([y1, act(x).unsqueeze(-1)], -1).unsqueeze(-1)
        # [batch, in, k+1, 1]
        
        ctx.save_for_backward(x, t, w, i_full, slice2, slice3, w2, y2)
        
        return torch.sum(w2 * y2, dim=(1,2))
        
    @staticmethod
    def backward(ctx, D_out):
        x, t, w, i_full, slice2, slice3, w2, y2 = ctx.saved_tensors
        k, act = ctx.k, ctx.act
        batch_dim = x.shape[0]
                
        Dw = torch.zeros_like(w)
        # Add the last 'row' of w
        slice1 = torch.cat([i_full[..., :k], 
                            torch.full_like(i_full[..., :1], -1)], 
                           -1).unsqueeze(-1)
        D_out = torch.reshape(D_out, [batch_dim, 1, 1, -1])
        Dw[slice1, slice2, slice3] += (y2 * D_out)
        
        # Use F.grad with vmap to compute per sample gradients
        Dy1 = b_spline_diff(x, i_full, t, k) # (batch, in, k)
        Dy1 = torch.cat([Dy1, vmap(grad(act))(x.reshape(-1))
                        .reshape_as(x).unsqueeze(-1)], -1).unsqueeze(-1)
        # [batch, in, k+1, 1]
        
        Dx = torch.sum(D_out * w2 * Dy1, dim=(2,3))
        
        return Dx, Dw, None, None, None
# %%