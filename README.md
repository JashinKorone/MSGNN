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
month={May},
day={21},
abstract={Infectious disease forecasting has been a key focus and proved to be crucial in controlling epidemic. A recent trend is to develop forecasting models based on graph neural networks (GNNs). However, existing GNN-based methods suffer from two key limitations: (1) current models broaden receptive fields by scaling the depth of GNNs, which is insufficient to preserve the semantics of long-range connectivity between distant but epidemic related areas. (2) Previous approaches model epidemics within single spatial scale, while ignoring the multi-scale epidemic patterns derived from different scales. To address these deficiencies, we devise the Multi-scale Spatio-temporal Graph Neural Network (MSGNN) based on an innovative multi-scale view. To be specific, in the proposed MSGNN model, we first devise a novel graph learning module, which directly captures long-range connectivity from trans-regional epidemic signals and integrates them into a multi-scale graph. Based on the learned multi-scale graph, we utilize a newly designed graph convolution module to exploit multi-scale epidemic patterns. This module allows us to facilitate multi-scale epidemic modeling by mining both scale-shared and scale-specific patterns. Experimental results on forecasting new cases of COVID-19 in United State demonstrate the superiority of our method over state-of-arts. Further analyses and visualization also show that MSGNN offers not only accurate, but also robust and interpretable forecasting result. Code is available at https://github.com/JashinKorone/MSGNN.},
issn={1573-756X},
doi={10.1007/s10618-024-01035-w},
url={https://doi.org/10.1007/s10618-024-01035-w}
}

```
