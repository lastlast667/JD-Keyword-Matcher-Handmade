"""
数据清洗 + 标注

使用方式：
    python run_clean_label.py

流程：
    1. 加载 data/raw/ 下最新的爬虫数据
    2. 调用 core.clean.clean() 去重+过滤
    3. 调用 core.label.infer_label() 打标签
    4. 过滤非技术岗，统计分布
    5. 保存到 data/intermediate/jobs_labeled.json
"""
from debugpy.launcher import output

from config.settings import RAW_DATA_DIR,INTERMEDIATE_DATA_DIR
from core.clean import clean
from core.label import infer_label
import json
from pathlib import Path
from collections import Counter

def step_clean():
    """
    清洗数据
    """
    print("="*50)
    print("step 1: 清洗数据")
    print("="*50)

    # 查找最新的爬虫数据
    files = sorted(list(RAW_DATA_DIR.glob("shixiseng_*.json")))
    if not files:
        raise FileNotFoundError(f"未找到爬虫数据文件，请检查 {RAW_DATA_DIR} 目录")

    latest_data_path = files[-1]
    print(f"加载数据: {latest_data_path}")

    # 加载数据
    with open(latest_data_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    # 去重+过滤
    cleaned_data = clean(jobs)

    stem = latest_data_path.stem
    cleaned_data_path = INTERMEDIATE_DATA_DIR / f"{stem}_cleaned.json"
    with open(cleaned_data_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"已保存到 {cleaned_data_path}")
    return cleaned_data_path


def step_label(cleaned_data_path:Path):
    """
    标注数据
    """
    print("="*50)
    print("step 2: 标注数据")
    print("="*50)

    # 直接用传入的路径加载清洗后的数据
    with open(cleaned_data_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"待标注数据量: {len(jobs)}")

    # 给每条数据打标签
    labeled_jobs = []
    for job in jobs:
        labels = infer_label(job["title"])
        job["category"] = labels["train"]
        job["sub_category"] = labels["display"]
        labeled_jobs.append(job)

    print(f"已标注数据量: {len(labeled_jobs)}")
    # 过滤非技术岗
    tech_jobs = [job for job in labeled_jobs if job["category"] != "过滤"]
    print(f"技术岗位数据量: {len(tech_jobs)}，非技术岗位数据量: {len(labeled_jobs) - len(tech_jobs)}")

    # 统计分布
    category_count = Counter([job["category"] for job in tech_jobs])
    print("技术岗位分布：")
    for category, count in category_count.most_common():
        print(f"{category}: {count}")

    # 保存到文件
    stem = cleaned_data_path.stem
    labeled_data_path = INTERMEDIATE_DATA_DIR / f"{stem}_labeled.json"
    with open(labeled_data_path, "w", encoding="utf-8") as f:
        json.dump(labeled_jobs, f, ensure_ascii=False, indent=2)

    print(f"已保存到 {labeled_data_path}（{len(tech_jobs)}条）")
    return labeled_data_path

def main():
    """
    主函数
    """
    print("="*50)
    print("JD关键词智能匹配系统 - 数据清洗+标注")
    print("="*50)
    # step 1: 清洗数据
    cleaned_data_path = step_clean()
    if not cleaned_data_path.exists():
        raise FileNotFoundError(f"清洗后的数据文件不存在: {cleaned_data_path}")
    # step 2: 标注数据
    labeled_data_path = step_label(cleaned_data_path)
    if not labeled_data_path.exists():
        raise FileNotFoundError(f"标注后的数据文件不存在: {labeled_data_path}")

    # 完成
    print("="*50)
    print("数据清洗+标注完成")
    print(f"已保存到 {labeled_data_path}")
    print("="*50)

if __name__ == "__main__":
    main()