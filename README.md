### Video Inpainting by HSV Color Space Guided Wavelet Convolution
# Overview
![圖片](https://github.com/Jhh8856/Wavelet_Unit/assets/42765536/9b36d7e3-8a77-412c-b1ae-6a26cc9cd3e4)

# Requirement
```
python == 3.7
opencv-python == 4.4.0.46
tensorflow == 2.4.1
Pillow
numpy
```
# Clone Repo
```
git clone https://github.com/Jhh8856/Wavelet_Unit
```
# Quick test
```
Python sample.py
```
# Params
1. `batchsize` in WaveletUnit.py and sample.py must be same.
2. `samp_seed` in sample.py must lower then dataset minus images for training.
3. Change dataset path in `load_model()`(line 49 in sample.py)
4. `train_count` and `test_count` in sample.py must divisible by `batchsize`
5. `masked` in sample.py doesn't work in free from generated mask mode

# To Do
Some function about Video Inpainting
Dataset format example
