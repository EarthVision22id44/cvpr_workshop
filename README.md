# Simultaneous Assimilation and Downscaling of simulated Sea Surface Heights with Deep Image Prior


This repository provides code to reproduce results from the paper [Simultaneous Assimilation and Downscaling of simulated Sea Surface Heights with Deep Image Prior
](https://).

The training setup is as in the following diagram:

<img src="https://github.com/EarthVision22id44/cvpr_workshop/blob/master/figures/DIP4DVarSR.png" width="400">

Requirements
---
- python==3.7.5 
- torch==1.6.0
- numpy==1.18.2
- optuna==2.10.0
- matplotlib==3.3.2

Simulate data - Observations from a Shallow water model
---
run ./data/simulate_data.py

Run Data Assimilation demo
---
in the jupyter notebook ./notebook_demo/1.Data_Assimilation, you can run assimilation examples with 4DVar, Reg. 4DVar and DIP 4DVar algorithms
You can choose: the downscaling factor r in [1,2,4,8], the level of noise noise_percent, and the sample index i corresponding to a trajectory in the database.

Run Main experiments
---
- run ./main_4DVar.py
- run ./main_DIP.py
