# +
import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
import json

def save_split_point_results(config, split_point_acc):
    # 保存 split_point_acc 数组
    with open(config.split_acc_path, 'w', encoding='utf-8') as f:
        json.dump(split_point_acc, f, ensure_ascii=False, indent=2)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(range(config.range_start, config.range_end, config.range_step),
             split_point_acc, marker='o', linestyle='-', color='b', label="Test Accuracy")
    plt.xlabel("Split Point")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Test Accuracy vs Split Point n={config.n} step = {config.step}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.plot_path)
    plt.close()

    print(f"✅ 保存结果和图像至: {config.saved_result_dir}")

def save_config_to_file(config):
    with open(config.config_path, 'w', encoding='utf-8') as f:
        for attr, value in sorted(vars(config).items()):
            f.write(f"{attr.upper()} = {value}\n")

