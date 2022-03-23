# Graph-based Embedding Smoothing (GES)

This is our Tensorflow implementation for the paper:

>Tianyu Zhu, Leilei Sun, and Guoqing Chen. "Graph-based Embedding Smoothing for Sequential Recommendation." IEEE Transactions on Knowledge and Data Engineering (2021).

## Introduction
Graph-based Embedding Smoothing (GES) is a general framework for improving sequential recommendation methods with sequential and semantic item graphs.

![](https://github.com/zhuty16/GES/blob/master/framework.jpg)

## Citation
```
@article{zhu2021graph,
  title={Graph-based Embedding Smoothing for Sequential Recommendation},
  author={Zhu, Tianyu and Sun, Leilei and Chen, Guoqing},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2021},
  publisher={IEEE}
}
```

## Environment Requirement
The code has been tested running under Python 3.6. The required packages are as follows:
* tensorflow == 1.5.0
* numpy == 1.14.2
* scipy == 1.1.0

## Dataset
[Google Drive](https://drive.google.com/drive/folders/1ny_jqRE_NwK3SbnxF4W3Ql_SiItKLKlC?usp=sharing)

## Example to Run the Codes
* Amazon Books dataset
```
python main.py --dataset=Amazon
```

