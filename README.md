# Code For Minimal Gated Unit(MGU)

## This repository contains code and some experiments for Minimal Gated Unit. For any problem concerning the code, please feel free to contact Mr. Chen-Lin Zhang (zhangcl@lamda.nju.edu.cn). 
## This packages are free for academic usage. You can run them at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com)

## Operating System:
####  Ubuntu Linux 14.04
## Requirements:
####  Python2.7 (Anaconda is preferred)
####  Theano
####  Lasagne
####  GPU with CUDA Support (Optional)

#### Hints: The code is originally developed in Theano v0.7 and Lasagne v0.1. But I test it with Theano v0.9 and Lasagne v0.2dev1. It's also OK.

## The MGU code is in MGULayer.py. other contains some utility code for performing the experiments.

## Perform the experiments in the paper
#### First, you should install the lasagne (http://lasagne.readthedocs.io/en/latest/user/installation.html) on your computer.
#### This repository contains three experiments: adding problems, mnist and imdb problem. For each problem, you can enter the folder and run
```
python IRNN_gru_2014.py
```
#### to perform the experiment with the GRU layer.
#### and you can run
```
python IRNN_gru_2015.py
```
#### to perform the experiment with our MGU layer.
