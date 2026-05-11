"""
classifier.py
===========
岗位分类模型训练

流程：
1. 加载 preprocessor.py 产出的 jobs_processed.json + tfidf_vectorizer.pkl
2. 构造 X（TF-IDF 特征）和 y（train_label）
3. 标签编码（字符串标签 → 整数）
4. 训练集/测试集划分（stratify=y 保证类别比例一致）
5. 两个模型对比：MultinomialNB vs LogisticRegression
6. 5 折交叉验证，macro-F1 评估
7. 选出最佳模型，在测试集上输出详细 classification_report
8. 保存：best_classifier.pkl + label_encoder.pkl

设计决策说明：
- 为什么用 macro-F1：10 个类别样本量不均衡（29~700+），macro 对每个类别同等看待，
  能反映模型在小类别上的表现；accuracy 会被大类别主导。
- 为什么用 StratifiedKFold：保持每折中各类别比例与整体一致，避免某折缺失小类别。
- 为什么对比 NB 和 LR：NB 是文本分类 baseline，训练快；LR 可解释性强，
  且能通过 coef_ 看出每个特征对各类别的贡献权重。
"""

# ================== 配置区 ==================

from config.settings import PROCESSED_DATA_DIR,MODEL_DIR
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import classification_report,f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np

# ================== step1 加载数据 ==================
def load_data():
    """
    加载处理后的数据和tfidf向量化器
    :return:
    """
    processed_data_file = sorted(list(PROCESSED_DATA_DIR.glob("shixiseng_*_processed.json")))
    tfidf_vectorizer_file = sorted(list(MODEL_DIR.glob("shixiseng_*_tfidf_vectorizer.pkl")))
    if not processed_data_file:
        raise FileNotFoundError(f"未找到处理后数据文件，请先运行 preprocessor.py")
    if not tfidf_vectorizer_file:
        raise FileNotFoundError(f"未找到 TF-IDF 向量化器文件，请先运行 preprocessor.py")

    last_processed_data_file = processed_data_file[-1]
    last_tfidf_vectorizer_file = tfidf_vectorizer_file[-1]

    # 加载预处理数据
    with open(last_processed_data_file, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
    # 加载 TF-IDF 向量化器
    with open(last_tfidf_vectorizer_file, "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    stem = last_processed_data_file.stem
    base_name = stem.rsplit("_",1)[0]

    return processed_data,tfidf_vectorizer,base_name,stem

# ================== step2 构造 X 和 y ==================
def construct_X_y(processed_data: list[dict],tfidf_vectorizer: TfidfVectorizer) -> tuple[list[str], list[str]]:
    """
    从原始数据构造模型所需的 X 和 y

    关键决策：
    即使 json 里存了 tokenized_words，我们仍然用 vectorizer.transform() 现场算 X，
    确保训练和推理的特征生成流程完全一致。
    """
    # y：加载原始字符串标签
    y_raw = [item['category'] for item in processed_data]
    # X：TF-IDF 特征
    X = tfidf_vectorizer.transform([item['tokens'] for item in processed_data])
    return X,y_raw

# ================== step3 标签编码 ==================
def encode_labels(y_raw: list[str]) -> tuple[list[str], LabelEncoder, list[str]]:
    """
    把字符串标签编码为整数，保存 encoder 供推理时解码
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    label_names = list(label_encoder.classes_)

    print(f"\n类别数量：{len(label_names)}")
    dist = Counter(y_raw)   # 统计每个类别的数量
    for name in label_names:
        print(f"{name:25s} : {dist[name]:4d}")
    return y_encoded,label_encoder,label_names

# ================== step4 划分训练集/测试集 ==================
def split_train_test(X: list[str], y_encoded: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    划分训练集/测试集，stratify=y 保证类别比例一致
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    return X_train, X_test, y_train, y_test

# ================== step5 模型训练 + 5折交叉验证对比 ==================
def compare_models(X_train: list[str], y_train: list[str]) -> tuple:
    """
    两个模型对比：MultinomialNB vs LogisticRegression
    5 折交叉验证，macro-F1 评估
    """

    candidates = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,                  # 最大迭代次数
            solver="lbfgs",                 # 优化算法
            C=0.5,                          # 正则化系数的导数，默认 1.0，小于1容易欠拟合，大于1容易过拟合
            # 新版本multi_class默认值是auto，会根据y的类型自动选择
            # multi_class="multinomial",    # 多分类
            random_state=42,                # 随机种子
            class_weight="balanced",        # 平衡类别权重
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    result = {}                 # 存储每个模型的平均 F1 分数
    best_score = -1             # 最佳模型的平均 F1 分数
    best_name = None            # 最佳模型的名称
    best_model = None           # 最佳模型对象

    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro",n_jobs=-1)

        result[name] = {
            "scores": scores,               # 5 折交叉验证分数
            "mean_score": scores.mean(),    # 平均分数
            "std_score": scores.std()       # 标准差
        }

        print(f"\n{name}")
        print(f"每折得分：{scores}")
        print(f"平均分数：{scores.mean():.4f}")
        print(f"标准差：{scores.std():.4f}")
        print(f"平均F1：{scores.mean():.4f} （+/- {scores.std():.4f}）")

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_name = name
            best_model = model

    print(f"\n最佳模型：{best_name} （交叉验证 macro-F1 = {best_score:.4f}）")

    # 用最佳模型在全量训练集上训练，得到最终模型
    best_model.fit(X_train, y_train)

    return best_model,result
# ================== step6 选出最佳模型 ==================
def select_best_model(best_model, X_test: list[str], y_test: list[str],label_names: list[str]) -> tuple:
    """
    选出最佳模型，在测试集上输出详细 classification_report
    """

    y_pred = best_model.predict(X_test)                     # 预测标签

    macro_f1 = f1_score(y_test, y_pred, average="macro")    # 宏 F1 分数
    acc = (y_test == y_pred).mean()                         # 准确率

    print("=" * 50)
    print(f"\n测试集评估结果：")
    print("=" * 50)
    print(f"准确率：{acc:.4f}")
    print(f"macro-F1：{macro_f1:.4f}")
    print(f"分类报告：\n{classification_report(y_test, y_pred, target_names=label_names,digits=4)}")

    return macro_f1

# ================== step7 模型评估 ==================
def diagnose_overfittng(best_model,X_train: list[str], y_train: list[str],X_test: list[str], y_test: list[str],label_names: list[str]):
    """
    模型过拟合诊断
    """
    y_train_pred = best_model.predict(X_train)  # 在训练集上预测
    y_test_pred = best_model.predict(X_test)    # 在测试集上预测

    train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")   # 在训练集上计算宏 F1 分数
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")      # 在测试集上计算宏 F1 分数
    gap = train_macro_f1 - test_macro_f1

    print("=" * 50)
    print(f"\n过拟合诊断（macro-F1）：")
    print("=" * 50)
    print(f"训练集 macro-F1：{train_macro_f1:.4f}")
    print(f"测试集 macro-F1：{test_macro_f1:.4f}")
    print(f"差距：{gap:.4f}")
    print("差距越小，模型越好；macro-F1越大，模型越好")

    if gap < 0.05:
        print(f"拟合良好")
    elif gap < 0.12:
        print(f"轻微过拟合")
    else:
        print(f"严重过拟合，请尝试降低模型复杂度")

    return gap

# ================== step8 保存模型 ==================
def save_model(base_name, best_model: object,label_encoder: LabelEncoder):
    """
    保存最佳模型和标签编码器
    :param best_model:
    :param label_encoder:
    :return:
    """

    BEST_MODEL_DIR = MODEL_DIR / f"{base_name}_best_model.pkl"          # 最佳模型文件路径
    LABEL_ENCODER_DIR = MODEL_DIR / f"{base_name}_label_encoder.pkl"    # 标签编码器文件路径

    with open(BEST_MODEL_DIR, "wb") as f:
        pickle.dump(best_model, f)

    with open(LABEL_ENCODER_DIR, "wb") as f:
        pickle.dump(label_encoder, f)

    model_size = BEST_MODEL_DIR.stat().st_size / 1024
    encode_size = LABEL_ENCODER_DIR.stat().st_size / 1024

    print("模型已保存：")
    print(f"{BEST_MODEL_DIR}模型大小：{model_size:.1f} KB")
    print(f"{LABEL_ENCODER_DIR}标签编码器大小：{encode_size:.1f} KB")

# ================== step9 搜索最佳 C ==================
def search_best_C(X_train, y_train, X_test, y_test, label_names):
    """
    C 值网格搜索：看过拟合差距 + 测试集 macro-F1 + 小类别表现
    """

    C_candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 8.0]

    print(f"{'C':>6s} {'λ=1/C':>8s} {'CV-F1':>8s} {'Test-F1':>8s} {'Train-F1':>8s} {'Gap':>8s} {'小类F1>0':>8s}")
    print("-" * 70)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5折交叉验证

    for C in C_candidates:
        model = LogisticRegression(
            max_iter=1000,
            C=C,
            solver="lbfgs",
            random_state=42,
            class_weight='balanced',
        )
        # 交叉验证，得到5折平均F1，即5折交叉验证分数
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        # 在全量训练集上训练模型
        model.fit(X_train, y_train)
        # 训练集/测试集评估
        train_f1 = f1_score(y_train, model.predict(X_train), average="macro")
        test_f1 = f1_score(y_test, model.predict(X_test), average="macro")
        gap = train_f1 - test_f1

        # 小类别 F1（4个小类的平均）
        report = classification_report(
            y_test, model.predict(X_test),
            target_names=label_names, output_dict=True
        )
        small_classes = ["C++后端开发", "Python/Go后端开发", "深度学习/NLP工程师", "计算机视觉工程师"]
        small_f1_avg = np.mean([
            report.get(cls, {}).get("f1-score", 0)
            for cls in small_classes
        ])
        small_ok = sum(1 for cls in small_classes if report.get(cls, {}).get("f1-score", 0) > 0)

        print(
            f"{C:6.2f} {1 / C:8.2f} {cv_scores.mean():8.4f} {test_f1:8.4f} {train_f1:8.4f} {gap:8.4f} {small_ok:>4d}/4")

def main():
    print("="*50)
    print("岗位分类模型训练")
    print("="*50)

    # 1. 加载数据和tfidf_vectorizer向量化器
    print("\n[1/8] 加载数据和tfidf_vectorizer向量化器")
    processed_data, tfidf_vectorizer ,base_name ,stem= load_data()

    # 2. 构造 X 和 y
    print("\n[2/8] 构造 X 和 y")
    X,y_raw = construct_X_y(processed_data,tfidf_vectorizer)

    # 3. 标签编码
    print("\n[3/8] 标签编码")
    y_encoded,label_encoder,label_names = encode_labels(y_raw)

    # 4. 划分训练集/测试集
    print("\n[4/8] 划分训练集/测试集")
    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

    # 5. 模型对比 + 5折交叉验证
    print("\n[5/8] 模型对比")
    best_model,result = compare_models(X_train, y_train)

    # 6. 选出最佳模型
    print("\n[6/8] 选出最佳模型")
    select_best_model(best_model, X_test, y_test,label_names)

    # 7. 模型评估
    print("\n[7/8] 模型评估")
    diagnose_overfittng(best_model,X_train, y_train,X_test, y_test,label_names)

    # 8. 保存模型
    print("\n[8/8] 保存模型")
    save_model(base_name, best_model,label_encoder)

    # 9. 网格搜索
    print("\n[9/8] 网格搜索，排查综合最优的C值")
    search_best_C(X_train, y_train, X_test, y_test, label_names)

if __name__ == "__main__":
    main()
