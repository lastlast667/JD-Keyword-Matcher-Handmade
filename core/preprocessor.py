"""
预处理器
jieba分词 + 停用词过滤
"""
import jieba
from config.settings import DATA_DIR,MODEL_DIR,PROCESSED_DATA_DIR,INTERMEDIATE_DATA_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json
from utils.decorators import load_words_from

@load_words_from("stopwords.txt")
def load_stopwords(stopwords: list[str]) -> list[str]:
    """
    加载停用词表
    """

    return stopwords

@load_words_from("customwords.txt")
def load_customwords(customwords: list[str]) -> list[str]:
    """
    加载自定义词表
    """

    return customwords

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

def preprocess():
    """
    文本预处理：分词 + 过滤停用词
    TF-IDF向量化
    输入：job_labeled.json
    输出：jobs_preprocessed.json + tfidf_vectorizer.pkl
    """

    # 查找最新的标注数据
    file = sorted(list(INTERMEDIATE_DATA_DIR.glob("shixiseng_*_cleaned_labeled.json")))
    if not file:
        raise FileNotFoundError(f"未找到标注数据文件，请检查 {INTERMEDIATE_DATA_DIR} 目录")

    latest_data_path = file[-1]
    print(f"加载数据: {latest_data_path}")

    # 加载数据
    with open(latest_data_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    # ===== 新增：过滤掉"过滤"类别，不参与模型训练 =====
    jobs = [j for j in jobs if j.get("category") != "过滤"]
    print(f"已过滤'过滤'类别，剩余{len(jobs)}条技术岗数据")
    # ================================================

    stopwords = load_stopwords()  # 加载停用词表
    customwords = load_customwords()  # 加载自定义词表
    for word in customwords:
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

    stem = latest_data_path.stem
    base_name = stem.rsplit("_",2)[0]

    # 保存向量化结果
    tfidf_path = PROCESSED_DATA_DIR / f"{base_name}_processed.json"
    with open(tfidf_path, "w",encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"已保存TF-IDF向量化结果到 {tfidf_path}")

    # 保存向量化器，即文本特征提取器
    vectorizer_path = MODEL_DIR / f"{base_name}_tfidf_vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:  # "wb"保存的是二进制文件，“w"保存的是文本文件
        pickle.dump(vectorizer, f)
    print(f"已保存TF-IDF向量化器到 {vectorizer_path}")

    # 返回处理后的数据，向量化器，向量化矩阵
    return jobs, vectorizer, tfidf_matrix




if __name__ == "__main__":
    preprocess()
