# Infrared and Visible Image Fusion using Deep Learning (PyTorch Implementation)

This repository is based on the paper "Infrared and Visible Image Fusion using a Deep Learning Framework" by Li et al. and presents a project where the original MATLAB implementation has been converted to PyTorch. Using PyTorch, this project combines visible (VIS) and infrared (IR) images to generate high-quality fused images.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0c88026e-e6cb-4122-a311-cd4b81ca2c1c" alt="PyTorch Image Fusion Overview">
</p>

##  Introduction

본 프로젝트는 PyTorch로 구현된 딥러닝 프레임워크를 통해 적외선 및 가시광선 이미지를 결합하여 더 나은 정보 전달력을 제공하는 이미지 융합 결과를 생성합니다.

### Input

- **Visible Image**
<p align="center">
  <img src="https://github.com/user-attachments/assets/9fea99e4-18e8-4082-b9a0-8de8e869c8f9" alt="VIS1 Image" width="400">
</p>

- **IR Image**
<p align="center">
  <img src="https://github.com/user-attachments/assets/ea0abef9-e285-46a6-ad06-5cecd628b67a" alt="IR1 Image" width="400">
</p>

### Result

- **Fusiom Result**
<p align="center">
  <img src="https://github.com/user-attachments/assets/52a43162-ddc6-4933-b580-fcef5a05f722" alt="Fusion Result" width="400">
</p>

To run the project, simply execute the main.py file. Make sure to correctly specify the image paths for the input images in the code, and the program will generate the fused output.



##  Citation

*Li H, Wu X J, Kittler J. Infrared and Visible Image Fusion using a Deep Learning Framework[C]//Pattern Recognition (ICPR), 2018 24rd International Conference on. IEEE, 2018: 2705 - 2710.*

```
@inproceedings{li2018infrared,
  title={Infrared and Visible Image Fusion using a Deep Learning Framework},
  author={Li, Hui and Wu, Xiao-Jun and Kittler, Josef},
  booktitle={2018 24th International Conference on Pattern Recognition (ICPR)},
  pages={2705--2710},
  year={2018},
  organization={IEEE}
}
