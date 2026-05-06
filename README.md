# JD关键词智能匹配系统

## 项目目标
基于机器学习的岗位描述（JD）智能分类与简历匹配系统。

## 技术栈
- 爬虫：Playwright（Boss直聘）
- 数据处理：pandas + jieba
- 机器学习：scikit-learn（TF-IDF + 朴素贝叶斯/逻辑回归）
- 展示：Streamlit

## 项目结构
D:.
|   .gitignore
|   app.py
|   directory_tree.txt
|   README.md
|   requirements-dev.txt
|   requirements.txt
|   
+---.idea
|   |   .gitignore
|   |   JD-Keyword-Matcher-Handmade.iml
|   |   misc.xml
|   |   modules.xml
|   |   vcs.xml
|   |   workspace.xml
|   |   
|   \---inspectionProfiles
|           profiles_settings.xml
|           Project_Default.xml
|           
+---config
|       settings.py
|       
+---core
|       classifier.py
|       matcher.py
|       preprocessor.py
|       __init__.py
|       
+---data
|   +---processed
|   |       .gitkeep
|   |       
|   \---raw
|           .gitkeep
|           
+---models
|       __init__.py
|       
+---spiders
|       shixiseng.py
|       __init__.py
|       
+---tests
|       __init__.py
|       
\---utils
        __init__.py

## 快速开始
```bash
pip install -r requirements.txt
python spiders/boss_zhipin.py
streamlit run app.py