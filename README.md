# HeLo

This repository contains the code of ACM-MM 2025 paper "HeLo: Heterogeneous Multi-Modal Fusion with Label Correlation for Emotion Distribution Learning".

## Contributions

![Network Architecture](https://github.com/kaio-99/HeLo/blob/main/figs/model.png)

- Due to differences in heterogeneity across modalities, a cross-attention mechanism is adopted to fuse the physiological data. Then, an optimal transport (OT)-based heterogeneity mining module is devised to effectively fuse the physiological and behavioral representations.
- For the learning of label correlation, we introduce a learnable label embedding, which is constrained by its learnable label correlation and ground-truth label correlation. Furthermore, the learnable label embeddings and label correlation are integrated through a novel label correlation-driven cross-attention mechanism.

## Recommended System Configurations

* numpy==2.1.3
* scikit_learn==1.4.2
* scipy==1.14.1
* torch==1.12.1+cu116

## Datasets

- [DMER](https://doi.org/10.5281/zenodo.11194571)
- [WESAD](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)

## Usage

Training the model by running train.py, and generating the test results via test.py. 

## Citation

If you use this code, please cite the corresponding paper:

```
@inproceedings{10.1145/3746027.3754852,
author = {Zheng, Chuhang and Tian, Chunwei and Wen, Jie and Zhang, Daoqiang and Zhu, Qi},
title = {HeLo: Heterogeneous Multi-Modal Fusion with Label Correlation for Emotion Distribution Learning},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3754852},
doi = {10.1145/3746027.3754852},
abstract = {},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {5519–5527},
location = {Dublin, Ireland},
series = {MM '25}
}
```
