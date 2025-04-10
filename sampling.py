# +
import os
import re
import sys
import jieba
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

# 你自己定义的模块
from config import *

# +
import sys
import contextlib

def suppress_print(func):
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(None):
            return func(*args, **kwargs)
    return wrapper


# +
@suppress_print
def load_names(config):
    name_path = os.path.join(config.data_dir,"names.txt")
    names = open(name_path, encoding='utf-8')
    names = names.read().split("　")
    names = list(names)
    for name in names:
        if len(name) == 3:
            names.append(name[1:])
    return names

import re
import jieba
import pandas as pd
import random
from collections import defaultdict
from config import LSTMConfig



def get_vocab(config,df):
    names=load_names(config)
    
    # 文本预处理 生成vocab
    vocab = []
    for i in df.index.tolist():
        chap = df.iloc[i, 1]
        chap = chap.split('。')
        for line in chap:
            temp = jieba.lcut(line)
            words = []
            for j in temp:
                # 过滤掉所有的标点符号
                j = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", j)
                if len(j) > 0:
                    if j in names:
                        j = '人名'
                    words.append(j)
            if len(words) > 0:
                vocab.append(words)
    return vocab

def get_text(config,df):
    names=load_names(config)
    # 生成样本
    y = []
    lines = []
    #config.step = 30
    #config.n = 40
    # 标记样本属于哪一回
    c = []
    config.s1 = int(config.split_point / 60 * config.step)
    config.s2 = int((120 - config.split_point) / config.split_point * config.s1)

    for i in df.index.tolist():
        f = df.iloc[i, 1]
        sp = df.iloc[i, 0]
        temp = jieba.lcut(f)
        words = []
        l = 0
        for j in temp:
            j = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", j)
            if j in names:
                j = '人名'
            if len(j) > 0:
                words.append(j)
        while l + config.n <= len(words):
            lines.append(list(words[l:l + config.n]))
            if sp <= config.split_point:
                y.append(1)
                l += config.s1
                c.append(sp)
            else:
                y.append(0)
                l += config.s2
                c.append(sp)

    return lines, y, c

import random

def oversample_to_balance(lines, y, c):
    # 统计正负样本数量
    pos_samples = [lines[i] for i in range(len(y)) if y[i] == 1]
    neg_samples = [lines[i] for i in range(len(y)) if y[i] == 0]
    pos_chapters = [c[i] for i in range(len(y)) if y[i] == 1]
    neg_chapters = [c[i] for i in range(len(y)) if y[i] == 0]

    pos_count = len(pos_samples)
    neg_count = len(neg_samples)
    

    # 找到需要过采样的类别
    if pos_count > neg_count:
        # 过采样负样本
        extra_samples = random.choices(neg_samples, k=pos_count - neg_count)
        extra_chapters = random.choices(neg_chapters, k=pos_count - neg_count)

        neg_samples.extend(extra_samples)
        neg_chapters.extend(extra_chapters)

    elif neg_count > pos_count:
        # 过采样正样本
        extra_samples = random.choices(pos_samples, k=neg_count - pos_count)
        extra_chapters = random.choices(pos_chapters, k=neg_count - pos_count)

        pos_samples.extend(extra_samples)
        pos_chapters.extend(extra_chapters)

    # 合并样本
    new_lines = pos_samples + neg_samples
    new_y = [1] * len(pos_samples) + [0] * len(neg_samples)
    new_c = pos_chapters + neg_chapters

    return new_lines, new_y, new_c
# -


