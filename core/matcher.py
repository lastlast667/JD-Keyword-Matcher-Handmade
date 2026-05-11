"""
简历-JD 相似度匹配引擎

核心流程（二次匹配策略）：
1. 加载预训练模型（TF-IDF 向量化器 + 分类器 + 标签编码器）
2. 接收用户简历文本（PDF / Word / 纯文本）
3. 对简历做与训练数据完全一致的分词 + TF-IDF 向量化
4. 分类器预测简历所属的大类（train_label）
5. 在该大类的 JD 池内，逐条计算 cosine similarity
6. 返回 Top-K 最相似的 JD 及匹配度分数

为什么用二次匹配？
- 模型只负责"粗分"（大类），cosine 负责"精排"（具体 JD）
- 避免训练一个 50+ 细分类别的模型（样本稀疏、效果差）
- 大类预测错了怎么办？→ 可以扩展为"预测大类 + 相邻大类"多池搜索

关键约束：
- 简历的分词流程必须与训练数据完全一致（同一个 tokenize 函数）
- 特征维度必须一致（用同一个 TfidfVectorizer transform，不能 fit）
"""

import pickle
import json
from config.settings import MODEL_DIR,PROCESSED_DATA_DIR
from core.preprocessor import tokenize,load_stopwords
from pathlib import Path
from utils.extract_text import extract_pdf_text,extract_docx_text,extract_doc_text
import numpy as np
from scipy.sparse import issparse

# ================== 路径配置 ==================


# ================== step 1:加载所有预训练资产 ==================
def load_artifacts():
    """
    加载三个 pickle 文件 + 预处理的 JD 数据

    Returns:
        vectorizer: TfidfVectorizer（只 transform，不复 fit），训练好的文本特征提取器（向量化器）
        classifier: LogisticRegression（预测大类），机器学习分类模型
        encoder: LabelEncoder（把整数标签转回字符串），训练好的标签编码器
        processed_data: list[dict] 所有带 category 的 JD 数据
    """
    best_model_file = sorted(list(MODEL_DIR.glob("shixiseng_*_best_model.pkl")))[-1]
    label_encoder_file = sorted(list(MODEL_DIR.glob("shixiseng_*_label_encoder.pkl")))[-1]
    tfidf_vectorizer_file = sorted(list(MODEL_DIR.glob("shixiseng_*_tfidf_vectorizer.pkl")))[-1]
    processed_data_file = sorted(list(PROCESSED_DATA_DIR.glob("shixiseng_*_processed.json")))[-1]

    if not best_model_file:
        raise FileNotFoundError("模型文件不存在")
    if not label_encoder_file:
        raise FileNotFoundError("标签编码器文件不存在")
    if not tfidf_vectorizer_file:
        raise FileNotFoundError("TF-IDF 向量化器文件不存在")
    if not processed_data_file:
        raise FileNotFoundError("JD 数据文件不存在")

    with open(best_model_file, "rb") as f:
        classifier = pickle.load(f)
    with open(label_encoder_file, "rb") as f:
        encoder = pickle.load(f)
    with open(tfidf_vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)
    with open(processed_data_file, "r", encoding="utf-8") as f:
        processed_data = json.load(f)

    print(f"加载模型: {best_model_file}")
    print(f"加载标签编码器: {label_encoder_file}")
    print(f"加载TF-IDF 向量化器: {tfidf_vectorizer_file}")
    print(f"加载预处理的JD数据: {processed_data_file}")

    # 打印基本信息
    print("\n" + "=" * 50)
    print("✅ 所有预训练资产加载完成，基本信息如下：")
    print("-" * 50)

    # 1. JD数据信息
    print(f"JD 数据总量：{len(processed_data)}条")

    # 2. TF-IDF 向量化器信息
    print(f"TF-IDF 向量化器信息：")
    print(f"特征维度：{vectorizer.get_feature_names_out().shape[0]}")

    # 3. 标签编码器信息
    print(f"标签编码器信息：")
    print(f"标签数量：{len(encoder.classes_)}")
    print(f"标签列表：{encoder.classes_}")

    # 4. 分类器信息
    print(f"分类器信息：")
    print(f"模型类型：{classifier.__class__.__name__}")
    print(f"模型参数：{classifier.get_params()}")

    return vectorizer, classifier, encoder, processed_data

# ================== step 2:简历文本提取 ==================
def extract_resume_text(source) -> str:
    """
    从各种来源提取简历纯文本

    Args:
        source: 可以是以下任一类型
          - str: 直接是简历文本（Streamlit 粘贴输入）
          - Path / str 路径: PDF 或 Word 文件路径

    Returns:
        text: 简历纯文本字符串

    设计决策：
    - 先实现 str 路径的情况（从文件读取），纯文本输入是 Streamlit 层的事
    - PDF 推荐用 PyPDF2 或 pdfplumber（后者对中文简历提取效果更好）
    - Word 用 python-docx
    - 如果第三方库不可用，优雅降级为报错提示
    """
    # step 1: 统一转换为Path对象，同时处理str和Path两种输入
    try:
        # 如果是纯文本，直接返回
        if isinstance(source, str) and not Path(source).exists():
            return source

        # 转换为Path对象，统一处理路径
        file_path = Path(source)

        # 检查是否是文件
        if not file_path.is_file():
            raise ValueError(f"{file_path} 不是一个文件")

        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} 文件不存在")

        # step 2: 取后缀转小写判断，解决大小写问题
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return extract_pdf_text(file_path)
        elif suffix == ".docx":
            return extract_docx_text(file_path)
        elif suffix == ".doc":
            return extract_doc_text(file_path)
        else:
            # 如果不是以上三种格式，则尝试读取纯文本
            print(f"{file_path} 文件不是 PDF/Word 格式，尝试读取纯文本")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"{file_path} 文件读取失败：{e}")
    except Exception as e:
        print(f"简历文本提取失败：{e}")
        return

# ================== step 3:简历向量化 ==================
def vectorize_resume(text: str, vectorizer) -> list[float]:
    """
    把简历文本转为 TF-IDF 向量

    关键约束：
    - 用 vectorizer.transform()，不是 fit_transform()！
    - 输入必须是空格拼接的分词后字符串（和训练时格式一致）

    Returns:
        vec: sparse matrix (1, n_features) —— 单条样本也是二维矩阵
    """

    # 加载停用词表
    stopwords = load_stopwords()
    tokens = tokenize(text,stopwords)

    # 向量化
    resume_vec = vectorizer.transform([tokens])

    # 返回向量化矩阵
    return resume_vec

# ================== step 4:预测简历大类 ==================
def predict_category(resume_vec, classifier, encoder):
    """
    预测简历所属的大类

    Returns:
        label_str: 字符串标签，如 "Python/Go后端开发"
        label_id: 整数标签
        probs: 各类别概率数组（用于调试或展示置信度）
    """

    label_id = classifier.predict(resume_vec)
    label_str = encoder.inverse_transform([label_id])
    probs = classifier.predict_proba(resume_vec)

    print(f"预测结果: {label_str}")

    return label_str,label_id,probs

# ================== step 5:类别池内相似度匹配 ==================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度（纯numpy实现，让你看懂原理）
    公式：cosθ = (a·b) / (||a|| * ||b||)
    也就是：两个向量的点积 除以 两个向量的模长的乘积

    Args:
        a: 向量1，形状 (n_features,)
        b: 向量2，形状 (n_features,)

    Returns:
        similarity: 余弦相似度分数，范围 [0, 1]（因为TF-IDF向量都是非负数）
    """

    # 1. 如果是稀疏矩阵，先转成稠密数组
    if issparse(a):
        a = a.toarray()
    if issparse(b):
        b = b.toarray()

    # 2. 如果是二维数组（形状 (1, n_features)），转成一维数组
    if a.ndim == 2:
        a = a.flatten()
    if b.ndim == 2:
        b = b.flatten()

    # 计算点积：对应位置相乘再相加
    dot_product = np.dot(a, b)

    # 计算两个向量的模长（L2范数）
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 防止除以零（如果两个向量都是零向量，相似度为0）
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 计算最终相似度
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def match_in_category_pool(resume_vec, processed_data, tfidf_vectorizer, target_label, top_k=10) -> list[dict]:
    """
    在目标大类的 JD 池内，逐条计算 cosine similarity，返回 Top-K

    Args:
        resume_vec: (1, n_features) 简历 TF-IDF 向量
        processed_data: list[dict] 所有 JD 数据（每个 dict 含 category 和 tokens）
        target_label: str, 目标大类标签
        top_k: int, 返回前 K 条

    Returns:
        matches: list[dict]，每条包含 JD 信息和 similarity 分数

    关键决策：
    - 为什么逐条算 cosine，不预先把类别池 stack 成矩阵一次性算？
      → 类别池通常几百条，逐条计算 overhead 极小；但代码更直观易懂。
      → 如果追求极致性能，可以预先把每个类别的向量 stack 好存为 np.array。
    - similarity 值域是 [-1, 1]，TF-IDF 非负所以实际是 [0, 1]。
      但值通常不会很高（0.1~0.3 就算很相似了），这个要在展示层跟用户说明。
    """

    # 筛选出目标大类的 JD 池
    target_pool = [data for data in processed_data if data['category'] == target_label]

    # ✅ 必须显式返回空列表！不能只打印警告
    if not target_pool:
        print(f"⚠️  警告：类别 '{target_label}' 下没有找到任何JD数据")
        return []

    # 逐条处理：tokens转vec -> 计算cosine -> 存储
    matches = []
    for data in target_pool:
        tokens = data['tokens']                                         # 取出tokens（已是空格拼接的字符串）
        data_vec = tfidf_vectorizer.transform([tokens]).toarray()[0]    # 向量化

        # 计算cosine
        similarity = cosine_similarity(resume_vec, data_vec)

        # 存储
        result = {
            "content": data['content'],                 # 存储JD原始内容
            "similarity": round(similarity,4),          # 存储相似度，保留4位小数
            "category": data['category'],               # 存储JD所属大类
            "sub_category": data['sub_category'],       # 存储JD所属小类
            "company": data['company'],                 # 存储JD所属公司
            "title": data['title'],                     # 存储JD标题
            "academic":data['academic'],                # 存储JD所属学历
            "url":data['url']                           # 存储JD链接
        }

        matches.append(result)

    match_sorted = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:top_k]

    print(f"匹配结果: {match_sorted}")

    return match_sorted

# ================== step 6:调用接口 ==================
def find_matches(resume_path, top_k=10,multi_pool=False):
    """
    对外暴露的统一入口

    Args:
        resume_source: 简历文本(str) 或 文件路径(str/Path)
        top_k: 返回匹配数量
        multi_pool: 是否启用多池搜索

    Returns:
        list[dict]: Top-K 匹配的 JD
    """

    # step 1:加载模型和数据
    vectorizer, classifier, encoder, processed_data = load_artifacts()
    # step 2:提取简历文本
    resume_text = extract_resume_text(resume_path)
    # step 3:简历向量化
    resume_vec = vectorize_resume(resume_text, vectorizer)
    # step 4:预测简历大类
    category,category_id,probs = predict_category(resume_vec, classifier, encoder)
    # step 5:类别池内相似度匹配
    matches = match_in_category_pool(resume_vec, processed_data, vectorizer, category, top_k)
    return matches
# ================== main:本地测试入口 ==================
def main():
    # 用一条模拟简历测试
    test_resume = """
        张三
        教育背景：某大学计算机科学与技术本科
        技能：Python, Django, Flask, MySQL, Redis, RESTful API 设计
        项目经验：
        1. 某电商后台管理系统（Python + Django + MySQL）
           - 负责用户模块、订单模块的 RESTful API 开发
           - 使用 Redis 缓存热点数据，QPS 从 200 提升到 1500
        2. 爬虫数据抓取平台（Scrapy + MongoDB）
           - 设计分布式爬虫架构，日抓取量 50 万条
        实习经历：某互联网公司后端开发实习生
        """

    print("=" * 50)
    print("简历-JD 匹配测试")
    print("=" * 50)

    results = find_matches(test_resume, top_k=5)
    if not results:
        print("❌ 没有找到匹配的岗位")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n第{i}名（相似度：{r['similarity']:.2%}）")   # 百分比格式
            print(f"内容：{r['content']}...")

if __name__ == "__main__":
    main()