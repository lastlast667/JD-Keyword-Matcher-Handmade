"""
预处理器
jieba分词 + 停用词过滤
"""
import jieba
from config.settings import DATA_DIR,MODEL_DIR,PROCESSED_DATA_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

def load_stopwords():
    """
    加载停用词表
    """
    stopwords_file = DATA_DIR / "stopwords.txt" # 停用词表路径

    stopwords = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords

def load_custom_words():
    """
    加载自定义词表
    """
    customwords_file = DATA_DIR / "customwords.txt" # 自定义词表路径

    custom_words = []
    with open(customwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            custom_words.append(line.strip())
    return custom_words

def tokenize(text: str,stopwords:list) -> str:
    """
    批量添加到jieba分词器中
    :param text: 文本
    :return: 分词结果
    """

    # 分词
    tokens = jieba.lcut(text)
    # 过滤：非停用词 + 长度大于1
    filtered_tokens = [
        token for token in tokens
        if token not in stopwords and len(token) > 1
    ]

    return " ".join(filtered_tokens)

def preprocess(jobs):
    """
    文本预处理：分词 + 过滤停用词
    输入：job_labeled.json
    输出：jobs_preprocessed.json + tfidf_vectorizer.pkl
    """
    from pathlib import Path
    from config.settings import DATA_DIR,MODEL_DIR,PROCESSED_DATA_DIR

    stopwords = load_stopwords()  # 加载停用词表
    custom_words = load_custom_words()  # 加载自定义词表
    for word in custom_words:
        jieba.add_word(word)

    # 提取文本，分词 + 过滤停用词
    texts = []
    for j in jobs:
        combined = j.get("title","") + " " + j.get("content","")    # 合并标题和内容
        tokens = tokenize(combined,stopwords) # 分词 + 过滤停用词
        j["tokens"] = tokens    # 更新j，字符串
        texts.append(tokens)    # 保存分词结果

    print(f"分词完成，共{len(texts)}条数据")

    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        max_features=3000,  # 控制词汇表大小
        ngram_range=(1, 2), # 控制词组合
        min_df=2,           # 过滤低频词
        max_df=0.9          # 过滤高频词
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF向量化完成，共{tfidf_matrix.shape[0]}条数据，{tfidf_matrix.shape[1]}个特征")

    # 保存向量化结果
    tfidf_path = PROCESSED_DATA_DIR / "jobs_processed.json"
    with open(tfidf_path, "w",encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"已保存TF-IDF向量化结果到 {tfidf_path}")

    # 保存向量化器
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:  # "wb"保存的是二进制文件，“w"保存的是文本文件
        pickle.dump(vectorizer, f)
    print(f"已保存TF-IDF向量化器到 {vectorizer_path}")

    return jobs, vectorizer




if __name__ == "__main__":
    import json
    from config.settings import INTERMEDIATE_DATA_DIR

    # 查找最新的标注数据
    file =sorted(list(INTERMEDIATE_DATA_DIR.glob("shixiseng_*_cleaned_labeled.json")))
    if not file:
        raise FileNotFoundError(f"未找到标注数据文件，请检查 {INTERMEDIATE_DATA_DIR} 目录")

    latest_data_path = file[-1]
    print(f"加载数据: {latest_data_path}")

    # 加载数据
    with open(latest_data_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    preprocess(jobs)
