import numpy as np

def umeyama_align(src: np.ndarray, dst: np.ndarray, with_scale=True): 
    """ src, dst: (N,3) point sets (src -> dst). Returns s, R, t s.t. dst ≈ s R src + t """ 
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    
    X = src - mu_src
    Y = dst - mu_dst
    
    # covariance
    Sigma = (Y.T @ X) / src.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    
    R = U @ S @ Vt
    
    if with_scale:
        var_src = (X**2).sum(axis=1).mean()
        s = np.trace(np.diag(D) @ S) / (var_src + 1e-12)
    else:
        s = 1.0
    
    t = mu_dst - s * (R @ mu_src)
    
    return s, R, t