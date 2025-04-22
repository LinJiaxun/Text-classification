# Project Structure
The repository follows a modular and well-organized structure to support clean development, easy maintenance, and reproducibility. Below is an overview of the main folders:

src/: Contains all core source code files.

•	data/ : Input data directory containing various literature text such as hongloumeng.csv, sanguoyanyi.csv

•	saved_result : Saved models, logs, test accuracy and its plots

•	config.py : Configuration class (paths, hyperparameters, training range, log setting)

•	main.py : Main training and evaluation script 

•	main_paralled.py : Parallel training script for speeding up the whole process 

•	model.py : LSTM model definition

•	sampling.py  : Sampling logic for split_point-based data generation

•	save.py : Functions to save models, logs, test accuracy and its plots

# Environment
All experiments were conducted on a server with the following environments:

Host: GN-A40-070

CPU: 2× Intel(R) Xeon(R) Platinum 8452Y, 72 cores total (36 cores per socket, no hyper-threading)

GPU: NVIDIA A40, 48 GB VRAM, CUDA 12.5

Memory: 503 GB RAM

OS: Linux (kernel 5.14.0)

Python Version: 3.10.12

NVIDIA Driver: 550.90.07

# Requirement
numpy

pandas

matplotlib

tqdm

jieba

argparse

json5

torch>=1.9.0 

gensim>=4.0.0

tensorboard

scikit-learn

tqdm

# Run Sample 
```
python main_paralled.py --n 50 --step 20 --range_start 30 --range_start 100 --range_step 5
```
