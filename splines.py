# %%
import torch

def _in_local_support(x, t_end, ta, tb):
    return torch.logical_or(torch.logical_and(ta <= x, x < tb),
                    torch.logical_and(x == tb, tb == t_end))
    
def _evaluate(k: int, t: torch.Tensor, i: torch.Tensor, x: torch.Tensor):
    if k==1:    
        ta = t[i]
        tb = t[i+1]
        return _in_local_support(x, t[-1:], ta, tb)
        
    else:
        ta = t[i]
        tb = t[i+k]
        
        flag = _in_local_support(x, t[-1:], ta, tb)
        
        ta1 = t[i+1]
        tb1 = t[i+k-1]
        
        y = _evaluate(k-1, t, i, x) * ((x-ta)/(tb1-ta))
        y = torch.nan_to_num(y, 0., 0., 0.)
        
        dy = _evaluate(k-1, t, i+1, x) * ((tb-x)/(tb-ta1))
        y += torch.nan_to_num(dy, 0., 0., 0.)
        
        return y*flag

def evaluate_diff(k, t, i, x):
    y = torch.zeros_like(x)
    
    dt = t[i+k-1] - t[i]
    y = _evaluate(k-1, t, i, x)/dt
    y = torch.nan_to_num(y, 0., 0., 0.)
    
    dt = t[i+k] - t[i+1]
    dy = _evaluate(k-1, t, i, x)/dt
    y += torch.nan_to_num(dy, 0., 0., 0.)
    
    return y*(k-1)

def _evaluate_idx(k, x, t0, dt, t_end):
    f = lambda x: (x - t0)/dt + (k-1)
    return torch.maximum(torch.minimum(f(x), f(t_end-dt)),
                         f(t0)).floor().int()
    
def evaluate_all(k, t, x):
    i_max = _evaluate_idx(k, x, t[:1], t[k:k+1]-t[k-1:k], t[-1:])
    i_full = [i_max-j for j in range(k-1, -1, -1)]
    y = [_evaluate(k, t, i, x) for i in i_full]
    
    return i_full, y

def evaluate_diff_idx(k: int, t: torch.Tensor, x: torch.Tensor,
                      i_full: list):
    return [evaluate_diff(k, t, i, x) for i in i_full]
# %%
