# +
from torch.utils import data
from config import *
from gensim.models import Word2Vec

@suppress_print
def train_word2vec(x):
    model = Word2Vec(x, vector_size=200, window=3, min_count=1, workers=12, sg=1)
    return model

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def stratified_split_by_chapter(lines, y, c, test_size=0.2, val_size=0.1, random_state=2024):
    """
    按章节划分训练集、验证集和测试集
    每一章的后20%为测试集，前80%为训练集，训练集中的后10%为验证集
    """

    df = pd.DataFrame({'lines': lines, 'y': y, 'chapter': c})

    train_set, test_set, val_set = [], [], []
    train_labels, test_labels, val_labels = [], [], []

    # 每一章（章节）单独进行划分
    for chapter in df['chapter'].unique():
        df_chapter = df[df['chapter'] == chapter].sort_index()  # 确保按顺序排列

        # 获取前80%作为训练集，后20%作为测试集
        train_size = int(len(df_chapter) * (1 - test_size))
        temp_train = df_chapter[:train_size]
        temp_test = df_chapter[train_size:]

        # 在训练集内再划分出验证集（后10%作为验证集）
        val_size_in_train = int(len(temp_train) * val_size)
        temp_val = temp_train[-val_size_in_train:]
        temp_train = temp_train[:-val_size_in_train]

        # 添加到各自的集合
        train_set.append(temp_train['lines'].tolist())
        train_labels.append(temp_train['y'].tolist())

        val_set.append(temp_val['lines'].tolist())
        val_labels.append(temp_val['y'].tolist())

        test_set.append(temp_test['lines'].tolist())
        test_labels.append(temp_test['y'].tolist())

    # 拼接所有章节的数据
    X_train = [item for sublist in train_set for item in sublist]
    y_train = [item for sublist in train_labels for item in sublist]

    X_test = [item for sublist in test_set for item in sublist]
    y_test = [item for sublist in test_labels for item in sublist]

    X_val = [item for sublist in val_set for item in sublist]
    y_val = [item for sublist in val_labels for item in sublist]

    return X_train, X_test, X_val, y_train, y_test, y_val

class HLMDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

import numpy as np
# 数据预处理 作为LSTM输入
@suppress_print
class Preprocess():
    def __init__(self, config,sentences): #首先定义类的一些属性
        self.w2v_path =config.w2v_path
        self.sentences = sentences
        self.config = config
        #self.sen_len = config.sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前训练好的word to vec 模型读进来
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size #embedding的维度就是训练好的Word2vec中向量的长度
    def add_embedding(self, word):
        # 把word（"<PAD>"或"<UNK>"）加进embedding，并赋予他一个随机生成的representation vector
        # 因为我们有时候要用到"<PAD>"或"<UNK>"，但它俩本身没法放到word2vec中训练而且它俩不需要生成一个能反应其与其他词关系的向量，故随机生成
        vector = torch.empty(1, self.embedding_dim)#生成空的
        torch.nn.init.uniform_(vector)#随机生成
        self.word2idx[word] = len(self.word2idx)#在word2idx放入对应的index
        self.idx2word.append(word)#在idx2word中放入对应的word
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)#在embedding_matrix中加入新的vector
    @suppress_print    
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的 dictionary
        # 制作一个 idx2word 的 list
        # 制作一个 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.key_to_index):
            #print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        print('')
        self.embedding_matrix = torch.tensor(np.array(self.embedding_matrix))

        # 将"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.config.sen_len: #多的直接截断
            sentence = sentence[:self.config.sen_len]
        else:                            #少的添加"<PAD>"
            pad_len = self.config.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.config.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子里面的字变成相对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            #print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把labels转成tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

from torch import nn
import torch
import torch.optim as optim


class LSTM_2_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_2_Net, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = not fix_embedding

        # Multiple convolution layers with different kernel sizes
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=4, padding=2)
        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(input_size=64 * 3, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.embedding(inputs)  # Shape: (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # Reshape for CNN (batch_size, embedding_dim, seq_len)

        # Apply multiple CNN layers and concatenate outputs
        x1 = self.pool(self.relu(self.conv1(x)))
        x2 = self.pool(self.relu(self.conv2(x)))
        x3 = self.pool(self.relu(self.conv3(x)))
        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate along channel dimension
        x = x.permute(0, 2, 1)  # Reshape back (batch_size, reduced_seq_len, 128 * 3)

        x, _ = self.lstm(x)  # Apply BiLSTM
        x = x[:, -1, :]  # Take the last hidden state
        x = self.classifier(x)  # Apply classifier
        return x

def evaluation(outputs, labels): #定义自己的评价函数，用分类的准确率来评价
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

@suppress_print
def training(train, valid, model,config):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameter total: {}, trainable: {}'.format(total, trainable))

    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,weight_decay=0.01)
    best_acc = 0

    for epoch in range(config.num_epochs):
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            #print(f"[DEBUG] type(inputs): {type(inputs)}, type(labels): {type(labels)}")
            inputs = inputs.to(config.device, dtype=torch.long)
            labels = labels.to(config.device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            correct = evaluation(outputs, labels)
            total_acc += (correct / config.batch_size)
            total_loss += loss.item()
            
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch+1, i+1, t_batch, loss.item(), correct*100/config.batch_size), end='\r')

        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # Validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(config.device, dtype=torch.long)
                labels = labels.to(config.device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / config.batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, f"{config.model_save_path}")
                print('Saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()

def testing(test_loader, model,device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()

    return ret_output
# -


