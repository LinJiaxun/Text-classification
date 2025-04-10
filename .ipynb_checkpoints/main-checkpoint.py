import os
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from config import *
from sampling import *
from model import *
from save import *
from gensim.models import Word2Vec


def main():
    config = LSTMConfig()
    logger = Logger(config)
    
    # 加载数据
    df = pd.read_csv(os.path.join(config.data_dir, "hongloumeng.csv"), encoding='utf-8')
    #df = df.iloc[80:120,:].reset_index(drop=True)
    vocab = get_vocab(config, df)
    w2v_model = train_word2vec(vocab)
    w2v_model.save(config.w2v_path)
    split_point_acc = []

    # 开始按章节遍历训练
    for i in tqdm(range(config.range_start, config.range_end, config.range_step)):
        logger.log(f"=== Split Point: {i} ===")
        config.split_point = i
        
        lines, y, c = get_text(config, df)
        lines, y, c = oversample_to_balance(lines, y, c)

        # 划分数据
        X_train, X_test, X_val, y_train, y_test, y_val = stratified_split_by_chapter2(lines, y, c)
        logger.log(f"训练集类别分布: {Counter(y_train)}，验证集类别分布: {Counter(y_val)}，测试集类别分布: {Counter(y_test)}")

        # 预处理
        # train
        preprocess = Preprocess(config,X_train)
        embedding = preprocess.make_embedding(load=True)
        X_train_tensor = preprocess.sentence_word2idx()
        y_train_tensor = preprocess.labels_to_tensor(y_train)

        # test
        preprocess = Preprocess(config,X_test)
        embedding_test = preprocess.make_embedding(load=True)
        X_test_tensor = preprocess.sentence_word2idx()
        y_test_tensor = preprocess.labels_to_tensor(y_test)

        # val
        preprocess = Preprocess(config,X_val)
        embedding_val = preprocess.make_embedding(load=True)
        X_val_tensor = preprocess.sentence_word2idx()
        y_val_tensor = preprocess.labels_to_tensor(y_val)

        # 构造模型
        model = LSTM_2_Net(
            embedding=embedding,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            fix_embedding=config.fix
        ).to(config.device)

        # 构造 Dataset & Loader
        train_loader = torch.utils.data.DataLoader(
            HLMDataset(X_train_tensor, y_train_tensor),
            batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

        val_loader = torch.utils.data.DataLoader(
            HLMDataset(X_val_tensor, y_val_tensor),
            batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        # 开始训练
        logger.log("start training")
        training(train_loader, val_loader, model, config)
        logger.log("start testing")
        test_dataset = HLMDataset(X=X_test_tensor, y=None)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config.batch_size,
                                                  shuffle=False,
                                                  num_workers = config.num_workers)
        model = torch.load(config.model_save_path, weights_only=False)
        outputs = testing(test_loader, model, config.device)


        # 计算测试集准确率
        tmp = pd.DataFrame({"id": [str(i) for i in range(len(X_test_tensor))], "label": outputs, "true": y_test_tensor})
        tmp['equal'] = abs(tmp['label']-tmp['true'])
        accuracy = (1 - sum(tmp['equal'])/len(tmp['equal']))*100
        logger.log(f'第{i}章为分界点，测试准确率：{accuracy}')
        split_point_acc.append(accuracy)


    # 保存 split_point_acc 和图像
    save_split_point_results(config, split_point_acc)

    # 保存参数配置
    save_config_to_file(config)

    logger.log("All process done.")

if __name__ == "__main__":
    main()



