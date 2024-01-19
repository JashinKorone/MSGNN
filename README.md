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
- The code is implemented on Python 3.7.10 with PyTorch 1.7.1, Ubuntu 22.04.1.

## 4. Citation
If you find this repository useful in your research, please consider citing:
```script
To be determined

```