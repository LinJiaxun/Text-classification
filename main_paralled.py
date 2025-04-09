import os
import torch
from torch import nn
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from config import *
from sampling import *
from model import *
from save import *
from multiprocessing import Pool, set_start_method
import pandas as pd
from collections import Counter
from config import LSTMConfig
import os
from filelock import FileLock


# +
def train_and_save_word2vec_model(vocab, w2v_path):
    w2v_model = train_word2vec(vocab)
    w2v_model.save(w2v_path)
    print(f"✅ Word2Vec model saved at {w2v_path}")

from datetime import datetime

def safe_log(message: str, log_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    lock_path = log_path + ".lock"
    print(full_message)
    with FileLock(lock_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")


# -


def run_single_split(split_point, shared_w2v_model_path,parallel_model_dir, log_path):
    config = LSTMConfig()
    config.split_point = split_point
    config.device = "cuda:0"
    config.w2v_path = shared_w2v_model_path
    config.model_save_path = os.path.join(parallel_model_dir, f"model_split_{split_point}.pt")
    
    # set seed
    #np.random.seed(config.seed)
    #torch.manual_seed(config.seed)
    #torch.cuda.manual_seed_all(config.seed)

    df = pd.read_csv(os.path.join(config.data_dir, "hongloumeng.csv"), encoding='utf-8')
    vocab = get_vocab(config, df)

    with FileLock(shared_w2v_model_path + ".lock"):
        w2v_model = Word2Vec.load(shared_w2v_model_path)

    lines, y, c = get_text(config, df)
    lines, y, c = oversample_to_balance(lines, y, c)
    X_train, X_test, X_val, y_train, y_test, y_val = stratified_split_by_chapter(lines, y, c)

    preprocess = Preprocess(config, X_train)
    embedding = preprocess.make_embedding(load=True)
    X_train_tensor = preprocess.sentence_word2idx()
    y_train_tensor = preprocess.labels_to_tensor(y_train)

    preprocess = Preprocess(config, X_test)
    embedding_test = preprocess.make_embedding(load=True)
    X_test_tensor = preprocess.sentence_word2idx()
    y_test_tensor = preprocess.labels_to_tensor(y_test)

    preprocess = Preprocess(config, X_val)
    embedding_val = preprocess.make_embedding(load=True)
    X_val_tensor = preprocess.sentence_word2idx()
    y_val_tensor = preprocess.labels_to_tensor(y_val)

    model = LSTM_2_Net(
        embedding=embedding,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        fix_embedding=config.fix
    ).to(config.device)

    train_loader = torch.utils.data.DataLoader(HLMDataset(X_train_tensor, y_train_tensor), batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(HLMDataset(X_val_tensor, y_val_tensor), batch_size=config.batch_size, shuffle=False, num_workers=0)

    training(train_loader, val_loader, model, config)

    test_loader = torch.utils.data.DataLoader(HLMDataset(X_test_tensor, None), batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = torch.load(config.model_save_path)
    outputs = testing(test_loader, model, config.device)

    tmp = pd.DataFrame({"label": outputs, "true": y_test_tensor})
    tmp['equal'] = abs(tmp['label'] - tmp['true'])
    acc = (1 - sum(tmp['equal']) / len(tmp['equal'])) * 100
    safe_log(f"Split {split_point} 完成，准确率: {acc:.2f}%", log_path)

    return split_point, acc


if __name__ == "__main__":
    set_start_method("spawn")
    config = LSTMConfig()
    splits = list(range(config.range_start, config.range_end, config.range_step))
    

    # 主进程统一创建保存目录
    os.makedirs(config.saved_result_dir, exist_ok=True)
    parallel_model_dir = os.path.join(config.model_save_path, "parallel_model")
    os.makedirs(parallel_model_dir, exist_ok=True)

    shared_w2v_model_path = os.path.join(parallel_model_dir, "shared_w2v.model")
    log_path = os.path.join(config.model_save_path, "parallel_log.txt")
    
    safe_log(config.print_params(),log_path)
    # 训练共享 Word2Vec
    df = pd.read_csv(os.path.join(config.data_dir, "hongloumeng.csv"), encoding='utf-8')
    vocab = get_vocab(config, df)
    train_and_save_word2vec_model(vocab, shared_w2v_model_path)

    # 多进程训练
    with Pool(processes=min(6, os.cpu_count())) as pool:
        results = pool.starmap(run_single_split, [(split, shared_w2v_model_path, parallel_model_dir,log_path) for split in splits])

    result_df = pd.DataFrame(results, columns=["split_point", "accuracy"])
    result_df = result_df.sort_values("split_point")
    result_df.to_csv(os.path.join(config.saved_result_dir, "parallel_results.csv"), index=False)

    split_point_acc = result_df["accuracy"].tolist()
    save_split_point_results(config, split_point_acc)
    save_config_to_file(config)

    safe_log("All process done.",log_path)



