"""
去重+过滤空值
"""

import json
from config.settings import RAW_DATA_DIR,INTERMEDIATE_DATA_DIR

def clean(jobs: list) -> list:
    seen = set()
    cleaned = []
    duplicates = [] # 记录被误判的可能重复项

    for j in jobs:
        title = j.get('title')
        content = j.get('content')
        company = j.get('company','')

        # 1. 去除空值
        if not title or not content:
            continue

        # 2. 用title + company做快速判断
        key = f'{title}|{company}'
        if key not in seen:
            seen.add(key)
            cleaned.append(j)
        else:
            # 3. 如果标题+公司重复，再判断描述是否相似
            content_prefix = content[:100].strip()
            for exiting in cleaned:
                if exiting['title'] == title and exiting.get('company','') == company:
                    existing_content_prefix = exiting['content'][:100].strip()
                    if content_prefix == existing_content_prefix:
                        break
            else:
                cleaned.append(j)
                duplicates.append(j)    # 记录一下，方便后续人工核对
    print(f"清洗：{len(jobs)} -> {len(cleaned)}条，其中{len(duplicates)}条是标题重复但描述不同的岗位")
    return cleaned

if __name__ == '__main__':
    input_file = RAW_DATA_DIR / "shixiseng_20260507_164909.json"
    with open(input_file, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    cleaned = clean(jobs)
    output_file = INTERMEDIATE_DATA_DIR / "shixiseng_20260507_164909_cleaned.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"清洗结果已保存到文件：{output_file}")
