import os 
import sys
import torch 

gpu_available = torch.cuda.is_available()
DEVICE = 'cuda' if gpu_available else 'cpu'
torch.set_default_tensor_type('torch.cuda.FloatTensor') if gpu_available else None

sys.path.append(os.path.join(os.path.dirname(__file__), 'DPIR'))

import cv2
import matplotlib.pyplot as plt

from libs.pnp import admm, deep_denoiser, modulo
from libs.utils import load_img, load_model

normalize  = lambda x: (x - torch.min(x,  dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / (torch.max(x, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - torch.min(x,  dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
tensor2img = lambda x: (x.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255).astype('uint8')
img2tensor = lambda x: torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
centering  = lambda x: x - torch.mean(x, dim=(-1, -2), keepdim=True)

# image parameters
img_size           = 1024
wrapping_threshold = 64
# recovert parameters
max_iters = 5
epsilon   = 0.1
_lambda   = 0.1
gamma     = 1.1

# Load image
img_path = os.path.join(".", "data", "kodim23.png")
img      = load_img(img_path, img_size)
img_t     =  img2tensor(img) / wrapping_threshold

# Modulo operation
modulo_t  = modulo(img_t + torch.rand_like(img_t) * 0.1, 1.0).to(DEVICE) 
img_t     = centering(img_t).to(DEVICE) 


# Unwrapping
model_name = 'drunet_color'
model_pool = 'model_zoo' 
model = load_model(model_name, model_pool)
model = model.to(DEVICE)

img_est = admm(modulo_t, deep_denoiser, model, max_iters=max_iters, epsilon=epsilon, _lambda=_lambda*epsilon, gamma=gamma)


# Visualize

plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow( tensor2img(normalize(img_t)) )
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow( tensor2img(normalize(modulo_t)) )
plt.title("Modulo Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow( tensor2img(normalize(img_est)) )
plt.title("Recovered Image")
plt.axis('off')

plt.show()
