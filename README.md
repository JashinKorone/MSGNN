# MSGNN
This is the repository for paper "MSGNN: Multi-scale Spatio-temporal Graph Neural Network for Epidemic Forecasting", accepted by Data Mining and Knowledge Discovery (DMKD) in 2024. 

## 1. Setup
- Install CUDA 10.1
- run `setup_py.sh` to install the necessary dependecies for python environments.


## 2. Usage
To run MSGNN, please follow the steps below:
```shell
cd ~/MSGNN/src
# Change directory to the source code folder
python data_utils.py 
# Pulling epidemic data from Google, CSSE. 
python run_models.py --forecast_date 2021-05-01 
# Running ensemble for MSGNN
python run_ensemble.py --forecast_date 2021-05-01
# Get the predicting results
# check '../outputs/2021-05-01_forecast.csv' for details
```

## 3. Note
- To run MSGNN, an NVIDIA GPU with at least 6GB memory is required.
- The code is implemented on a server with Intel Core i7 10700F, 32GB of RAM, an NVIDIA RTX 2070 SUPER and Ubuntu 22.04.1 LTS.  

## 4. Citation
If you find this repository useful in your research, please consider citing:
```script
@Article{Qiu2024,
author={Qiu, Mingjie
and Tan, Zhiyi
and Bao, Bing-Kun},
title={MSGNN: Multi-scale Spatio-temporal Graph Neural Network for epidemic forecasting},
journal={Data Mining and Knowledge Discovery},
year={2024},
month={Jul},
day={01},
volume={38},
number={4},
pages={2348-2376},
issn={1573-756X},
doi={10.1007/s10618-024-01035-w},
url={https://doi.org/10.1007/s10618-024-01035-w}
}

```
