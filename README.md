# Federated Learning and Split Learning with raspberry pie
This is for releasing the source code of the SRDS 2020 paper "End-to-End Evaluation of Federated Learning and Split Learning for Internet of Things".

If you find it is useful and used for publication. Please kindly cite our work as:

```
@inproceedings{gao2020end,
title={End-to-End Evaluation of Federated Learning and Split Learning for Internet of Things},
author={Gao, Yansong and Kim, Minki and Abuadbba, Sharif and Kim, Yeonjae and Thapa, Chandra and Kim, Kyuyeon and Camtepe, Seyit A and Kim, Hyoungshick and Nepal, Surya},
booktitle={The 39th International Symposium on Reliable Distributed Systems (SRDS)},
year={2020}}
```

## Helpful Link
* [Experiment video](https://www.youtube.com/watch?v=x5mD1_EA2ps)
* [Manual for installing pytorch on raspberry pie3](https://github.com/Minki-Kim95/Install-pytorch-on-RaspberryPi)
* [Paper](https://arxiv.org/abs/2003.13376)
* [Presentation](https://www.youtube.com/watch?v=l55krlN4FLA&t=69s)

## Description
This repository contains the implementations of various distributed machine learning models like Federated learning, split learning and ensemble learning

## Requirements(Desktop)
  * Python==3.6
  * PyTorch==1.5.1
  
## Requirements(Raspberry pie3)
  * Python==3.7
  * PyTorch==1.0.0

## Repository summary
  - `models` directory: has pre-processed training/testing data of MIT arrhythmia ECG database in `hdf5` format. If you want, you can upload another preprocessed train and test data here.
  - `federated_learning` directory: source codes of federated learning in `ipynb` and `.py` format
  - `split_learning` directory: source codes of split learning in `ipynb` and `.py` format
  - `ensemble_learning` directory: source codes of ensemble learning in `ipynb` and `.py` format
  
## How to use

### 1. Run client on desktop
you need to use `~client.ipynb` file

### 2. Run client on raspberry pie
you need to use `~client_rasp.ipynb` or `~client_rasp.py` file
If you run these files, you can see the temperature, memory usage of raspberry pie.

## Overall process

**set hyperparameters**
- set variable `users`, in server and client file
- set variable `rounds`, `local_epoch` or `epochs` of training

**Running code**
- Run the server code first
- After run server, run the clients

**input information**
- if you run the server, you can see the printed **ip address** of server
- when you run the client you need the enter **order of client** and **ip address**
- if there is no problem, training will be started
  
## Project members
[Gao Yansong](https://github.com/garrisongys), [Kim Minki](https://github.com/Minki-Kim95), [Abuadbba Sharif](https://github.com/SharifAbuadbba), Kim Yeonjae, [Thapa Chandra](https://github.com/chandra2thapa), [Kim Kyuyeon](https://github.com/kyuyeonpooh), Camtepe Seyit A, [Kim Hyoungshick](https://github.com/hyoungshick), Nepal Surya
