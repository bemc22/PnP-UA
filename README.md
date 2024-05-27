# Deep Plug-and-Play Algorithm for Unsaturated Imaging
[![DOI:10.1109/ICASSP48485.2024.10446495](https://zenodo.org/badge/DOI/10.1109/ICASSP48485.2024.10446495.svg)](https://doi.org/10.1109/ICASSP48485.2024.10446495)
## Abstract
Commercial sensors often suffer from overexposure in bright regions, leading to signal clipping and information loss because of saturation. Existing solutions involve either employing logarithmic irradiance response sensors or capturing multiple shots from different saturation levels. However, these approaches can be complex or rely on static scenes, limiting their effectiveness in fully addressing the saturation problem. A promising solution is the use of unsaturated sensors, also known as modulo cameras, which employ an array of self-reset pixels to wrap the signal when it reaches the saturation level. The resulting image exhibits a noisy and discontinuous shape, requiring an unwrapping algorithm to obtain a smooth and continuous representation of the scene. We propose a deep plug-and-play algorithm that combines model-based optimization with a deep denoiser. By leveraging the spatial correlation of the scene within the close solution of an unwrapping step, our approach successfully unwraps the continuous values while simultaneously reducing noise. Extensive evaluations show the superiority of our method compared to state-of-the-art unwrapping and unmodulo algorithms in terms of reconstruction quality.

## How to Use
We use the DRUNet as deep denoiser, the model can be download from the [official repository](https://github.com/cszn/DPIR/tree/master/model_zoo).
- drunet_color.pth for color images
- drunet_gray.pth  for grayscale images

## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as


```bib
@inproceedings{bacca2024deep,
  title={Deep Plug-and-Play Algorithm for Unsaturated Imaging},
  author={Bacca, Jorge and Monroy, Brayan and Arguello, Henry},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2460--2464},
  year={2024},
  organization={IEEE}
  url = {https://doi.org/10.1109/ICASSP48485.2024.10446495},
  doi = {10.1109/ICASSP48485.2024.10446495}
}
```
