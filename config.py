import argparse
import os
import torch
import argparse
import sys
from datetime import datetime


# +
class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("\nParameters:")
        for attr, value in sorted(vars(self).items()):
            prtf(f"{attr.upper()} = {value}")
        prtf("")

    def as_markdown(self):
        text = "| Name | Value |\n|------|-------|\n"
        for attr, value in sorted(vars(self).items()):
            text += f"| {attr} | {value} |\n"
        return text

class LSTMConfig(BaseConfig):
    def build_parser(self):
        parser = argparse.ArgumentParser("param")
        # 随机数参数（暂未用到）        
        parser.add_argument('--seed', type=int, default=2, help='random seed')

        # 基础路径
        parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), 'data'))
        parser.add_argument('--w2v_path', type=str, default=os.path.join(os.getcwd(), 'w2v_all.model'))
        parser.add_argument('--embedding_matrix', default='embedding_matrix.npy')

        # 模型结构参数
        parser.add_argument('--embedding_dim', type=int, default=200)
        parser.add_argument('--input_size', type=int, default=1)
        parser.add_argument('--hidden_size', type=int, default=200)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--fix', type = bool,default=True, help='fix embedding during training')

        # 数据处理参数
        parser.add_argument('--sen_len', type=int, default=40)
        parser.add_argument('--split_point', type=int, default=None)
        parser.add_argument('--s1', type=int, default=None)
        parser.add_argument('--s2', type=int, default=None)
        parser.add_argument('--n', type=int, default=40)
        parser.add_argument('--step', type=int, default=40)

        # 训练参数
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--grad_clip', type=float, default=5.0)# 暂未用到
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.5)


        # 训练循环区间
        parser.add_argument('--range_start', type=int, default=30)
        parser.add_argument('--range_end', type=int, default=110)
        parser.add_argument('--range_step', type=int, default=5)

        # 日志和设备
        parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        parser.add_argument('--exp_name', default=datetime.now().strftime("exp_%Y%m%d_%H%M%S"))

        return parser

    def __init__(self):
        parser = self.build_parser()
        if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
            args = parser.parse_args(args=[])
        else:
            args = parser.parse_args()

        super().__init__(**vars(args))
        # Word2Vec
        self.w2v_path = os.path.join(os.path.join(os.getcwd(), "w2v_all.model"))


        # 构造日志路径
        self.log_dir = os.path.join("logs", self.exp_name)
        self.tensorboard_log_dir = os.path.join(self.log_dir, "tensorboard")
        self.txt_log_path = os.path.join(self.log_dir, "train.log")

        # 构造保存路径
        self.saved_result_dir = os.path.join("saved_result", self.exp_name)
        self.model_save_path = os.path.join(self.saved_result_dir, "best_model.pth")
        self.split_acc_path = os.path.join(self.saved_result_dir, "split_point_acc.json")
        self.plot_path = os.path.join(self.saved_result_dir, "accuracy_vs_split_point.png")
        self.config_path = os.path.join(self.saved_result_dir, "config.txt")

        # 创建目录（在所有路径都定义好之后）
        #os.makedirs(self.log_dir, exist_ok=True)
        #os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        #os.makedirs(self.saved_result_dir, exist_ok=True)

        #self.print_params()# 打印参数设置


# +
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.config = config
        self.log_file = open(config.txt_log_path, 'w', encoding='utf8')
        self.writer = SummaryWriter(log_dir=config.tensorboard_log_dir)
        
    def log(self, text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{timestamp}] {text}"
        print(text)
        self.log_file.write(text + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        self.writer.close()

    def write_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)



# +
import sys
import contextlib

def suppress_print(func):
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(None):
            return func(*args, **kwargs)
    return wrapper
# -


