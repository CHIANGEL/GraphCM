## A Graph-Enhanced Click Model for Web Search (GraphCM)

### Introduction

This is the pytorch implementation of GraphCM proposed in the paper: [A Graph-Enhanced Click Model for Web Search. SIGIR 2021](https://dl.acm.org/doi/10.1145/3404835.3462895).

### Requirements

**NOTE**: The versions of torch-cluster, torch-scatter, torch-sparse, torch-spline-conv are strictly required for torch-geometric package. You can follow the installation instruction in the PyG official website: [torch-geometric 1.6.3](https://pytorch-geometric.readthedocs.io/en/1.6.3/).

- python 3.7
- pytorch 1.6.0+cu101
- torchvision 0.7.0+cu101
- torch-cluster 1.5.8
- torch-scatter 2.0.5
- torch-sparse 0.6.8
- torch-spline-conv 1.2.0
- torch-geometric 1.6.3
- tensorboardx 2.1

### Input Data Formats

After data pre-processing, we can put all the generated files into ```./data/dataset/``` folder as input files for GraphCM. Demo input files are available under the ```./data/demo/``` directory. 

The format of train & dev & test & label input files is as follows:

- Each line: ```<session id><tab><query id><tab>[<document ids>]<tab>[<vtype ids>]<tab>[<clicks infos>]<tab>[<relevance>]```

### Quick Start

We provide quick start command in ```./run.sh```. Note that input files that are related to graph modules are not provided in this repo. You can genenrate graph-related input files using data preprocess files in the ```./data_preprocess/``` fold.

### Citation

If you find the resources in this repo useful, please cite our work.

```
@inproceedings{lin2021graph,
  title={A Graph-Enhanced Click Model for Web Search},
  author={Lin, Jianghao and Liu, Weiwen and Dai, Xinyi and Zhang, Weinan and Li, Shuai and Tang, Ruiming and He, Xiuqiang and Hao, Jianye and Yu, Yong},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1259--1268},
  year={2021}
}
```