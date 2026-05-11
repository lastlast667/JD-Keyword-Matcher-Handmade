"""
app.py
======
Streamlit 展示入口：简历-JD 智能匹配系统

页面结构：
1. 侧边栏：上传简历文件（PDF/DOCX/DOC）或粘贴文本
2. 主区域：
   - 上传/输入简历
   - 点击"开始匹配"按钮
   - 展示预测的大类 + 各类别概率条形图
   - 展示 Top-K 匹配结果卡片（标题、公司、相似度、JD摘要）
3. 结果解读说明

部署注意：
- Streamlit Cloud 的 Python 版本要与本地一致（避免 greenlet 兼容问题）
- requirements.txt 不能包含 playwright（部署环境不需要爬虫）
"""



import sys
from pathlib import Path

import streamlit as st
import numpy as np

# 把项目根目录加入 sys.path，确保能 import core/
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.matcher import (extract_resume_text,vectorize_resume,
                          predict_category,match_in_category_pool,load_artifacts)

import tempfile



# 设置页面配置
st.set_page_config(
    page_title="简历匹配",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 初始化：加载模型（只执行一次，用 st.cache_resource 缓存）
# ============================================================
@st.cache_resource
def init_model():
    """
    用 st.cache_resource 确保模型只加载一次，不随用户交互重新加载
    """

    vectorizer, classifier, encoder, processed_data = load_artifacts()
    return vectorizer, classifier, encoder, processed_data

try:
    vectorizer, classifier, encoder, processed_data = init_model()
    model_ready = True
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.error("请检查模型文件是否存在，或联系管理员更新模型")
    model_ready = False
    st.stop()

# ============================================================
# 页面标题
# ============================================================
st.title("🔍 JD关键词智能匹配系统")
st.markdown("基于 TF-IDF + 逻辑回归的简历-岗位智能匹配")

# ============================================================
# 侧边栏：输入方式选择
# ============================================================
with st.sidebar:
    st.header("📄 简历输入")

    input_method = st.radio(
        "选择输入方式：",
        options=["上传简历", "粘贴简历"],
        index=0,
        help="上传简历：从本地选择简历文件；粘贴简历：直接粘贴简历内容",
    )

    resume_text = ""

    if input_method == "上传简历":
        resume_file = st.file_uploader(
            "上传简历文件，支持 PDF、DOCX、TXT 格式",
            type=["pdf", "docx", "doc", "txt"]
        )
        if resume_file:
            # 获取当前系统的临时目录
            temp_dir = Path(tempfile.gettempdir())
            # 在临时目录下创建和上传文件同名的临时文件
            temp_path = temp_dir / resume_file.name
            # 写入上传文件的二进制内容
            temp_path.write_bytes(resume_file.getvalue())
            # 提取简历文本
            resume_text = extract_resume_text(temp_path)
            # 删除临时文件
            temp_path.unlink()
            if resume_text:
                st.success("简历提取成功！")
                st.write("简历内容预览：")
                st.write(resume_text)
            else:
                st.error("简历提取失败，请检查文件格式或内容是否正确")
    else:
        # 粘贴简历
        resume_text = st.text_area(
            "粘贴简历内容",
            height=200,
            placeholder="请粘贴简历内容..."
        )

    # 匹配参数
    st.header("🔍 匹配参数")
    top_k = st.slider("返回结果数量",min_value=1,max_value=20,value=10,step=1)

    # 匹配按钮
    match_button = st.button(
        "开始匹配",
        type="primary",
        disabled=not resume_text or not model_ready
    )

# ============================================================
# 主区域：结果展示
# ============================================================
if match_button and resume_text and model_ready:

    with st.spinner("正在分析简历并匹配岗位..."):
        # 1. 简历向量化
        resume_vec = vectorize_resume(resume_text, vectorizer)
        # 2. 预测简历大类
        label_str,label_id,probs = predict_category(resume_vec, classifier, encoder)
        # 3. 在大类池中匹配
        matches = match_in_category_pool(resume_vec, processed_data, vectorizer, label_str, top_k)

    # ---------- 预测类别展示 ----------
    st.header(f"📌 预测岗位方向：{label_str}")

    # 各类别概率条形图
    prob_data = {}
    for i in range(len(encoder.classes_)):
        # 把ID转成类别名称
        category_name = encoder.inverse_transform([i])[0]
        # 取出对应的概率
        probability = probs[0][i]
        # 把"类别名称: 概率"存入字典
        prob_data[category_name] = probability

    # 按概率降序取前5个
    top5_probs = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True)[:5])

    st.subheader("类别置信度分布（Top-5）")
    for category_name, prob in top5_probs.items():
        bar_color = "🟢" if category_name == label_str else "⚪"
        st.progress(float(prob),text=f"{bar_color} {category_name}: {prob:.2%}")

    st.divider()

    # ---------- 匹配结果展示 ----------
    if not matches:
        st.error("未找到匹配的岗位")
    else:
        st.header(f"🔍 匹配结果（Top-{top_k}）")

        for i, match in enumerate(matches, 1):
            # 相似度分段着色
            sim = match['similarity']
            if sim >= 0.25:
                badge = "🟢 高度匹配"
            elif sim >= 0.15:
                badge = "🟡 较为匹配"
            elif sim >= 0.08:
                badge = "🟠 一般匹配"
            else:
                badge = "🔴 匹配度较低"

            # 卡片式分栏展示
            with st.container(border=True):  # 加上border=True，变成真正的卡片
                # 分成左右两栏，比例3:1（左边宽放内容，右边窄放相似度）
                col1, col2 = st.columns([3, 1])

                with col1:
                    # 岗位标题（带序号和链接）
                    title = match.get('title', '未知岗位')
                    job_url = match.get('url', '')
                    st.markdown(f"## {i}. [{title} 🔗]({job_url})")
                    # 公司和子类信息（小字灰色）
                    st.caption(f"💼 {match.get('company', '未知公司')} | 📂 {match.get('sub_category', '')}")

                    # JD内容摘要（只显示前150字，太长加省略号）
                    content = match.get("content", "")
                    # if len(content) > 150:
                    #     content = content[:150] + "..."
                    st.markdown(content)

                    # 额外显示你原来代码里的其他字段（学历要求）
                    st.caption(f"🎓 学历要求：{match.get('academic', '不限')}")

                with col2:
                    # 右侧显示相似度大数字和匹配度标签
                    st.metric("相似度", f"{sim:.2%}")
                    st.caption(badge)

            # 每个卡片之间加分隔线
            st.divider()

# ============================================================
# 使用说明（页面底部）
# ============================================================
with st.expander("📖 使用说明"):
    st.markdown("""
    **匹配原理：**
    1. **分词+向量化**：你的简历和数据库中的 JD 都经过 jieba 分词 + TF-IDF 向量化
    2. **大类预测**：逻辑回归模型预测简历属于哪个岗位方向
    3. **相似度排序**：在预测类别的 JD 池内，用余弦相似度逐条比对，取最相似的 Top-K

    **相似度解读：**
    - 🟢 ≥ 25%：高度匹配，强烈推荐
    - 🟡 15%~25%：较为匹配，值得一看
    - 🟠 8%~15%：一般匹配，相关性较弱
    - 🔴 < 8%：匹配度较低

    **注意：** TF-IDF 余弦相似度的绝对值通常不会很高（词汇重叠有限），
    即使 15% 也代表在同类岗位中相当相关。
    """)