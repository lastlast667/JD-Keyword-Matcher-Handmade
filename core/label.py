"""
数据标注模块
职责：根据岗位title推断训练标签 + 展示子标签
"""

def infer_label(title: str) -> dict:
    """
    :param title:
    :return: {"train":"训练标签","display":"展示子标签"}
    """
    t = title.lower()

    # ========= 第一层：过滤非技术岗位 ==========
    non_tech = ["录入", "标注", "审核", "打字", "复制", "粘贴", "客服", "文员",
                "统计员", "主播", "销售", "运营专员", "运营助理", "行政", "人事", "HR"]
    if any(k in t for k in non_tech):
        return {"train": "过滤", "display": "非技术岗"}

    # ========= 第二层：明确非开发技术岗（之前被"工程师"兜底捕获的）==========

    # 测试开发
    test_signals = ["测试工程师", "测试开发", "自动化测试", "QA", "质量保证", "性能测试"]
    if any(s in t for s in test_signals):
        return {"train": "测试/运维开发", "display": "测试开发"}

    # 运维/SRE/平台
    ops_signals = ["运维", "SRE", "devops", "平台工程师", "稳定性", "监控", "部署", "发布系统"]
    if any(s in t for s in ops_signals):
        return {"train": "测试/运维开发", "display": "运维/SRE"}

    # ========= Round 1: 前端开发（优先级最高，特征最独特）（统一训练，细分展示） ==========
    frontend_signals = ["前端", "vue", "react", "angular", "javascript", "html", "css",
                        "webpack", "小程序", "node", "typescript", "flutter", "移动端"]
    if any(s in t for s in frontend_signals):
        if "vue" in t:
            return {"train":"前端开发","display":"Vue前端开发"}
        elif "react" in t:
            return {"train": "前端开发", "display": "React前端开发"}
        else:
            return {"train": "前端开发", "display": "前端开发"}

    # ========= Round 2: 人工智能（细分训练，直接展示） ==========
    ai_signals = ["人工智能", "深度学习", "神经网络", "强化学习", "大模型",
                       "aigc", "模型训练", "计算机视觉", "cv", "自然语言", "nlp"]
    computer_vision_signals = ["计算机视觉", "cv", "图像", "目标检测", "ocr"]
    deep_learning_NLP_engineer_signals = ["深度学习", "自然语言", "nlp", "bert", "transformer", "文本", "cnn", "rnn", "lstm"]
    machine_learning_engineer_signals = ["机器学习", "特征工程", "用户画像"]
    if any(s in t for s in ai_signals):
        if any(s in t for s in computer_vision_signals):
            return {"train": "计算机视觉工程师", "display": "计算机视觉工程师"}
        elif any(s in t for s in deep_learning_NLP_engineer_signals):
            return {"train": "深度学习/NLP工程师", "display": "深度学习NLP工程师"}
        elif any(s in t for s in machine_learning_engineer_signals):
            return {"train": "算法工程师", "display": "机器学习工程师"}
        else:
            return {"train": "算法工程师", "display": "AI算法工程师"}

    # 推荐/搜索/广告算法（从"算法工程师"拆分）
    rec_signals = ["推荐算法", "搜索算法", "广告算法", "召回", "排序算法", "推荐系统"]
    if any(s in t for s in rec_signals):
        return {"train": "算法工程师", "display": "推荐搜索算法"}

    # ========== Round 3：数据分析（统一训练，挖掘并入，细分展示） ==========
    data_analysis_signals = ["数据分析", "bi", "商业分析", "数据分析师", "tableau", "powerbi", "可视化", "报表", "指标体系",
                              "经营分析", "策略分析", "用户分析", "产品分析", "数据挖掘", "风控模型", "预测模型", "时序分析", "异常检测"]
    analyst_signals = ["数据分析", "bi", "商业分析", "数据分析师", "tableau", "powerbi", "可视化", "报表", "指标体系",
                       "经营分析", "策略分析", "用户分析", "产品分析"]
    mining_signals = ["数据挖掘", "挖掘工程师", "风控模型", "预测模型", "时序分析", "异常检测"]
    if any(s in t for s in data_analysis_signals):
        if any(s in t for s in analyst_signals):
            return {"train": "数据分析", "display": "数据分析师"}
        elif any(s in t for s in mining_signals):
            return {"train": "数据分析", "display": "数据挖掘工程师"}

    # ========== Round 4：后端开发（兜底）（细分训练，直接展示）（大数据开发并入）（增加测试/运维的分流后更精准） ==========
    backend_signals = ["java", "python", "go", "golang", "php", "c++", "c/c++",
                       "后端", "服务端", "后台", "spring", "django", "flask",
                       "微服务", "分布式", "高并发"]
    bigdata_signals = ["etl", "数仓", "数据仓库", "hive", "spark", "hadoop", "flink", "kafka", "大数据开发", "数据平台"]
    Java_backend_signals = ["java", "spring"]
    Python_Go_backend_signals = ["python", "django", "flask", "go", "golang"]
    PHP_backend_signals = ["php"]
    C_backend_signals = ["c++", "c/c++"]
    # 嵌入式/硬件
    embed_signals = ["嵌入式", "单片机", "arm", "rtos", "固件", "硬件工程师", "fpga", "驱动开发"]
    if any(s in t for s in backend_signals):
        # 展示子标签
        if any(s in t for s in Java_backend_signals):
            return {"train": "Java后端开发", "display": "Java后端开发"}
        elif any(s in t for s in Python_Go_backend_signals):
            return {"train": "Python/Go后端开发", "display": "Python/Go后端开发"}
        elif any(s in t for s in C_backend_signals):
            return {"train": "C++后端开发", "display": "C++后端开发"}
        elif any(s in t for s in embed_signals):
            return {"train": "其他后端开发", "display": "嵌入式开发"}
        else:
            return {"train": "其他后端开发", "display": "其他后端开发"}

    # ========== 兜底：其他技术岗（不再含"工程师"就进后端） ==========
    # 含"工程师""开发"但没命中任何明确类别的 → 过滤
    return {"train": "过滤", "display": "其他技术岗"}

if __name__ == "__main__":
    test_titles = [
        "Java后端开发实习生",
        "Vue3前端开发工程师",
        "计算机视觉算法实习生",
        "NLP自然语言处理工程师",
        "数据仓库建模工程师",
        "数据分析师",
        "数据挖掘算法工程师",
        "Python爬虫开发",
        "ETL开发工程师",
        "用户运营专员",  # 应该被过滤
    ]
    for t in test_titles:
        result = infer_label(t)
        print(f"{t:30s} -> 训练标签: {result['train']:15s} 展示子标签: {result['display']:15s}")





