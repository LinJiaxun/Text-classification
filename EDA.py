# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 22:02:29 2025

整合代码：
1. 红楼梦相对词频 Top20 可视化（全文+前八十回+后四十回）与句子长度分布
2. 章节词汇多样性趋势分析
3. 统计 dynamic 和 static word length
4. 功能性词分析
5. 输出 vocab 中的分词总词数及唯一词数
6. 全文字数统计（不含标点）


@author: 15339
"""

import pandas as pd
import re
import jieba
import jieba.analyse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# 设置停用词（这里排除“人名”，因为我们在处理时已替换）
stopwords = ['人名']

# （可选）将文本转换为 CSV（如果尚未转换，可取消下面函数调用的注释）
def text2csv():
    path = '/Users/zxk/Desktop/5188/data/hongloumeng.txt'
    with open(path, 'r', encoding='UTF-8') as f:
        text = f.read()
    
    reg = "第[一二三四五六七八九十百]+回"
    chapters = re.split(reg, text)
    chapters = [ch for ch in chapters if len(ch) > 200]
    
    if len(chapters) != 120:
        print(f"⚠️ 警告：分割后章节数为 {len(chapters)}，应为 120，请检查正则表达式！")
    
    index = range(1, len(chapters) + 1)
    result = pd.DataFrame({"id": index, "text": chapters})
    result.to_csv('D:/study/ST5188/data/hongloumeng.csv', index=False)
    print("✅ 文本已成功转换为 CSV！")

def get_full_text():
 # 读取 CSV 文件，每行对应一回文本
    df = pd.read_csv('/Users/zxk/Desktop/5188/data/hongloumeng.csv', encoding='utf-8')
    
    # 文本预处理：按句号拆分，再分词
    vocab = []  # 存放所有句子的分词列表
    for i in df.index.tolist():
        chap = df.iloc[i, 1]
        sentences = chap.split('。')
        for sentence in sentences:
            temp = jieba.lcut(sentence)
            words = []
            for j in temp:
                # 过滤掉标点符号和其它非文字内容
                j = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", j)
                # 保留长度大于 1 且不在停用词列表中的词
                if len(j) > 1 and j not in stopwords:
                    words.append(j)
            if len(words) > 0:
                vocab.append(words)
    return df, vocab

def get_text():
    """
    预处理文本：
    1. 读取人名并对长度为 3 的人名增加去姓版本
    2. 读取 CSV 中每回文本，按句号拆分，进行 jieba 分词，并替换人名
    3. 返回原始 DataFrame 和所有句子的分词结果（vocab）
    """
    name_path = "/Users/zxk/Desktop/5188/data/names.txt"
    names = open(name_path, encoding='utf-8').read().split("　")
    names = list(names)
    # 注意：这里遍历副本，避免在循环中动态增加导致无限循环
    for name in names.copy():
        if len(name) == 3:
            names.append(name[1:])
    
    # 读取 CSV 文件，每行对应一回文本
    df = pd.read_csv('/Users/zxk/Desktop/5188/data/hongloumeng.csv', encoding='utf-8')
    
    # 文本预处理：按句号拆分，再分词
    vocab = []  # 存放所有句子的分词列表
    for i in df.index.tolist():
        chap = df.iloc[i, 1]
        sentences = chap.split('。')
        for sentence in sentences:
            temp = jieba.lcut(sentence)
            words = []
            for j in temp:
                # 过滤掉标点符号和其它非文字内容
                j = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", j)
                if j in names:
                    j = '人名'
                # 保留长度大于 1 且不在停用词列表中的词
                if len(j) > 1 and j not in stopwords:
                    words.append(j)
            if len(words) > 0:
                vocab.append(words)
    print("分词完成！")
    return df, vocab

def analyze_text(df):
    """
    对每一回文本做章节统计：
    - 句子数、字数（按句子拆分的字符总数）和唯一词汇数
    并绘制章节唯一词汇数趋势图
    """
    chapter_stats = []
    for chap_id in range(len(df)):
        chapter_text = df.iloc[chap_id, 1]
        sentences = chapter_text.split('。')
        # 这里 word_count 统计每句字符数之和（可根据需要改为分词后的总词数）
        word_count = sum(len(sentence) for sentence in sentences)
        unique_words = len(set(jieba.lcut(chapter_text)))
        chapter_stats.append({
            "chapter": chap_id + 1,
            "sentence_count": len(sentences),
            "word_count": word_count,
            "unique_words": unique_words
        })
    stats_df = pd.DataFrame(chapter_stats)
    
    # 绘制章节词汇多样性趋势图（唯一词汇数）
    plt.figure(figsize=(12, 6))
    plt.plot(stats_df['chapter'], stats_df['unique_words'], 
             color='#2c7bb6', linewidth=2, marker='o', markersize=4,
             markerfacecolor='white', markeredgewidth=1)
    plt.xlabel('Chapter Number', fontsize=12, labelpad=10) 
    plt.ylabel('Unique Words Count', fontsize=12, labelpad=10) 
    plt.title('Vocabulary Diversity Trend in Dream of the Red Chamber', 
              fontsize=14, pad=20) 
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xticks(range(0, 121, 20))  # 每 20 章一个刻度
    plt.xlim(1, 120)
    plt.legend(['Unique Words'])
    plt.tight_layout() 
    plt.show()
    return stats_df

def analyze_word_length(vocab, df):
    """
    计算 dynamic 和 static word length
    """
    # ---- Static Word Length Analysis ----
    word_lengths = [len(word) for sentence in vocab for word in sentence]
    print(f"平均词长: {np.mean(word_lengths):.2f}")
    print(f"词长中位数: {np.median(word_lengths)}")
    print(f"最短词长: {np.min(word_lengths)}, 最长词长: {np.max(word_lengths)}")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(word_lengths, bins=20)
    plt.xlabel("Word Length")
    plt.ylabel("Frequency")
    plt.title("Static Word Length Distribution")
    plt.show()
    
    # ---- Dynamic Word Length Analysis ----
    chapter_word_lengths = []
    for i in df.index.tolist():
        chap = df.iloc[i, 1]
        words = [word for word in jieba.lcut(chap) if len(word) > 1]
        avg_length = np.mean([len(word) for word in words]) if words else 0
        chapter_word_lengths.append(avg_length)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(df)+1), chapter_word_lengths, marker='o', linestyle='-', color='#2c7bb6')
    plt.xlabel("Chapter Number")
    plt.ylabel("Average Word Length")
    plt.title("Dynamic Word Length Across Chapters")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
    return chapter_word_lengths

# -------------------- 功能性词分析 --------------------
function_words = ["的", "了", "在", "是", "和", "与", "也", "但", "如果", "因为"]
def function_word_analysis(df, function_words):
    counts = {word: 0 for word in function_words}
    for text in df['text']:
        words = jieba.lcut(text)
        for word in words:
            if word in function_words:
                counts[word] += 1
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.xlabel("Function Words")
    plt.ylabel("Frequency")
    plt.title("Function Word Usage in Text")
    plt.show()
    
 
# -------------------- 全文字数统计（不含标点） --------------------
def count_words(text):
    text = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，\- 。？、~@#￥%……&*（）：；‘]+", "", text)
    return len(text)

# -------------------- 词汇相对频率分析 --------------------
def compute_relative_freq(vocab):
    all_words = [word for sentence in vocab for word in sentence]
    word_counts = Counter(all_words)
    total_word_count = len(all_words)
    rel_freq = {word: count / total_word_count for word, count in word_counts.items()}
    return sorted(rel_freq.items(), key=lambda x: x[1], reverse=True)[:20]

def plot_top20(common_words, title):
    words, freqs = zip(*common_words)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(words), y=list(freqs))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Words")
    plt.ylabel("Relative Frequency")
    plt.title(title)
    plt.show()
    
# -------------------- 句子长度统计与可视化 --------------------
def analyze_sentence_lengths(vocab, title):
    sentence_lengths = [len(sentence) for sentence in vocab]
    print(f"{title} - Average sentence length: {np.mean(sentence_lengths):.2f}")
    print(f"{title} - Median sentence length: {np.median(sentence_lengths)}")
    print(f"{title} - Shortest sentence length: {np.min(sentence_lengths)}, Longest sentence length: {np.max(sentence_lengths)}")
    
    plt.figure(figsize=(10, 5))
    sns.histplot(sentence_lengths, bins=30, kde=True)
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.title(f"{title} - Sentence Length Distribution")
    plt.xlim(0, 40)
    plt.show()

if __name__ == "__main__":
    # 如有需要先将文本转换为 CSV，请取消下面行的注释：
    # text2csv()
    
    # 获取预处理数据
    df, vocab = get_text()
    df_full, vocab_full = get_full_text()
    
    # -------------------- 词频统计与可视化 --------------------
    # 合并所有句子的分词结果
    all_words = [word for sentence in vocab for word in sentence]
    word_counts = Counter(all_words)
    print("最常见的 20 个词：", word_counts.most_common(20))
    
    # 输出 vocab 中的分词总词数与唯一词汇数
    total_word_count = len(all_words)
    unique_word_count = len(set(all_words))
    print(f"分词的总词数：{total_word_count}")
    print(f"分词的唯一词汇数：{unique_word_count}")
    
    #绘制top20相对词频
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    top20 = compute_relative_freq(vocab)
    plot_top20(top20, "Top 20 Relative Frequency Words")
    
    # 词汇相对频率分析（整体及分段对比）
    vocab_80 = vocab[:80]  
    vocab_40 = vocab[80:] 
    
    top20_80 = compute_relative_freq(vocab_80)
    top20_40 = compute_relative_freq(vocab_40)
    
    plot_top20(top20_80, "Top 20 Relative Word Frequencies in the First 80 Chapters")
    plot_top20(top20_40, "Top 20 Relative Word Frequencies in the Last 40 Chapters")
    
    # -------------------- 句子长度统计与可视化 --------------------
    analyze_sentence_lengths(vocab, "Full Text")
    analyze_sentence_lengths(vocab_80, "First 80 Chapters")
    analyze_sentence_lengths(vocab_40, "Last 40 Chapters")
    
    # -------------------- 章节词汇多样性分析 --------------------
    stats_df = analyze_text(df)
    
    
    chapter_word_lengths = analyze_word_length(vocab, df)
    
    # 功能性词分析
    function_word_analysis(df, function_words)
    
    
    total_chars = sum(count_words(chap) for chap in df_full.iloc[:, 1])
    print(f"Full text word count (excluding punctuation)：{total_chars}")
    
    
    # 打印章节统计摘要
    print("\n【统计摘要】")
    summary = stats_df.agg({
    'unique_words': ['mean', 'std'],
    'word_count': ['mean', 'std']
    })
    print(summary)
