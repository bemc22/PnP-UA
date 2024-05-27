import torch 
import torch_dct as dct
import torch.nn.functional as F

def modulo(x, mx):
    positive_input = x > 0
    x = x % mx
    x = torch.where((x == 0) & positive_input, torch.tensor(mx, dtype=x.dtype, device=x.device), x)
    return x

def wrapToMax(x, mx):
    return modulo(x + mx/2, mx) - mx/2


def deep_denoiser(x, noise_level=50, model=None):

    min_vals = x.min(dim=(-1), keepdim=True)[0].min(dim=(-2), keepdim=True)[0]
    max_vals = x.max(dim=(-1), keepdim=True)[0].max(dim=(-2), keepdim=True)[0]

    x_input = torch.clamp(x, min_vals, max_vals)
    x_input = (x - min_vals) / (max_vals - min_vals)

    noise_level_map = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) * noise_level / 255.
    x_input = torch.cat((x_input, noise_level_map), dim=1)

    output = model(x_input)

    output = output * (max_vals - min_vals) + min_vals

    return output


class Unwrapping(object):

    def __init__(self, y, mx):
        # Mdx_y = M( Delta_x @ y ) , Mdy_y = M( Delta_y @ y )
        Mdx_y = F.pad( wrapToMax(torch.diff(y, 1, dim=-1), mx), (1, 0), mode='constant')
        Mdy_y = F.pad( wrapToMax(torch.diff(y, 1, dim=-2), mx), (0, 0, 1, 0), mode='constant')

        # DTMDy = D^T ( Mdx_y, Mdy_y )
        DTMDy =  - ( torch.diff(F.pad(Mdx_y, (0, 1)), 1, dim=-1) + torch.diff(F.pad(Mdy_y, (0,0, 0, 1)), 1, dim=-2) )

        self.DTMDy = DTMDy
        self.mx = mx


    def forward(self, xtilde, epsilon):
        
        rho = self.DTMDy + (epsilon / 2) * xtilde
        dct_rho = dct.dct_2d(rho, norm='ortho')

        NX, MX = rho.shape[-1], rho.shape[-2]
        I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
        I, J = I.to(rho.device), J.to(rho.device)

        I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

        denom = 2 * ( ( epsilon / 4 ) + 2 - ( torch.cos(torch.pi * I / MX ) + torch.cos(torch.pi * J / NX ) ) )
        denom = denom.to(rho.device)

        dct_phi = dct_rho / denom
        dct_phi[..., 0, 0] = 0

        phi = dct.idct_2d(dct_phi, norm='ortho')
        return phi
    


def admm(y, denoiser, model, max_iters,  mx=1.0, epsilon=1.0, _lambda=1.0, gamma=1.0):
        

        # initialize variables
        unwrapping_fn = Unwrapping(y, mx)

        u_t = torch.zeros_like(y)

        # initialize x_t
        x_t = unwrapping_fn.forward(u_t, 0.0)

        # denoising step
        vtilde = x_t + u_t
        sigma = _lambda/epsilon
        v_t = denoiser(vtilde, sigma, model)

        # update step
        u_t = u_t + x_t - v_t

        for i in range(max_iters):
            epsilon = epsilon * gamma	

            # inversion step
            xtilde = v_t - u_t
            x_t =  unwrapping_fn.forward(xtilde, epsilon)

            # denoising step
            vtilde = x_t + u_t
            sigma = _lambda/epsilon
            v_t =  denoiser(vtilde, sigma, model)

            # update step
            u_t = u_t + x_t - v_t


        x_hat = v_t 
        x_hat = x_hat - torch.mean(x_hat, dim=(-1, -2), keepdim=True)
    
        return x_hat