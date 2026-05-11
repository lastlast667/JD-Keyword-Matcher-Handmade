"""
shixiseng.py
实习僧 Playwright 爬虫

设计思路：
    1. 最外层：遍历关键词列表
    2. 中层：每个关键词遍历页码（1~MAX_PAGES）
    3. 内层：列表页只提取20个详情页href，然后逐个goto抓取
    4. 保存：所有数据汇总到一个JSON

技术要点：
    - 有头模式(headless=False)用于首次调试，确认无误后切无头
    - 不模拟点击，直接提取<a>标签href，效率最高
    - 详情页用稳定选择器提取字段
"""

import json
import time
import random
from pathlib import Path
import traceback

from playwright.sync_api import sync_playwright

# ================== 配置区 ==================

# 调试开关
HEADLESS = True

# 延迟
# 格式：（最短数秒，最长数秒）
DELAY_BETWEEN_PAGES = (1, 2)    # 页面切换
DELAY_BETWEEN_DETAILS = (0.5, 1)    # 详情页抓取
DELAY_BETWEEN_KEYWORDS = (3, 5)    # 关键词切换

# 爬取页数
# 热门词会自动爬满20页，冷门词爬不满会自动停
MAX_PAGES_PER_KEYWORD = 100

# 输出路径
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

# 关键词分组（先定义四个方向，后面做分类用）
KEYWORDS = [
    # 批次1：后端开发
    'java','python','c','go','后端',
    # 批次2：前端开发
    'css','web','前端',
    # 批次3：人工智能
    '计算机视觉','深度','自然语言处理','算法','人工智能',
    # 批次4：数据类
    '数据','BI','ETL'
]

# URL模板：实习僧搜索页
# page=页码，keyword=关键词，type=实习，city=全国
LIST_URL_TEMPLATE = (
    "https://www.shixiseng.com/interns"
    "?page={page}"
    "&keyword={keyword}"
    "&type=intern"
    "&city=全国"
)

# 详情页选择器
DETAIL_SELECTOR = ".intern-wrap.interns-point.intern-item"

# ================== 核心函数 ==================

def extract_detail_url(page) -> list[str]:
    """
    在列表页提取当前页所有详情页URL
    """
    try:
        page.wait_for_selector(DETAIL_SELECTOR, timeout=15000)
    except Exception:
        return []  # 连卡片都等不到，说明这页没数据了

    # 找到所有卡片
    links = page.query_selector_all(DETAIL_SELECTOR)

    detail_urls = []
    for link in links:
        # 找到<a>标签，再提取href
        a_tag = link.query_selector("a[href]")
        if a_tag:
            detail_url = a_tag.get_attribute("href")
            if detail_url:
                detail_urls.append(detail_url)
    return detail_urls

def extract_detail(page, detail_url:str, keyword:str) -> dict | None:
    """
    进入详情页，提取完整字段

    Args:
        page: 详情页页面对象
        detail_url: 详情页URL
        keyword: 本次搜索的关键词（用于后续分类标注）

    returns:
        一个包含岗位信息的字典，提取失败返回None
    """
    try:
        # 直接goto详情页，比点击+go_back快很多
        page.goto(detail_url, timeout=15000,wait_until="domcontentloaded")

        # 等待关键元素出现（职位标题容器加载完成）
        page.wait_for_selector(".job-header", timeout=15000)

        # 提取字段
        # 职位标题
        title_el = page.query_selector(".new_job_name span")
        title = title_el.inner_text().strip() if title_el else None
        # 公司名称
        company_el = (page.query_selector(".com-name"))
        company = company_el.inner_text().strip() if company_el else None
        # 薪资
        salary_el = (page.query_selector(".job_money.cutom_font"))
        salary = salary_el.inner_text().strip() if salary_el else None
        # 地点
        location_el = (page.query_selector(".job_position"))
        location = location_el.inner_text().strip() if location_el else None
        # 学历
        academic_el = (page.query_selector(".job_academic"))
        academic = academic_el.inner_text().strip() if academic_el else None
        # 工作时间
        work_time_el = (page.query_selector(".job_week.cutom_font"))
        work_time = work_time_el.inner_text().strip() if work_time_el else None
        # 实习周期
        practice_period_el = (page.query_selector(".job_time.cutom_font"))
        practice_period = practice_period_el.inner_text().strip() if practice_period_el else None
        # 职位描述
        content_el = (page.query_selector(".job_detail"))
        content = content_el.inner_text().strip() if content_el else None

        # 返回结果
        return {
            "title": title,
            "company": company,
            "salary": salary,
            "location": location,
            "academic": academic,
            "work_time": work_time,
            "practice_period": practice_period,
            "content": content,
            "keyword": keyword, # 关键词用于后续分类标注
            "url": detail_url,  # 详情页链接，用于展示时跳转
        }

    except Exception as e:
        print(f"提取详情页失败: {e}")
        return None

def run_spider():
    """
    主爬虫流程
    """
    all_jobs = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.0 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.0"
            ),
        )
        list_page = context.new_page()

        # ======= 外层循环：关键词 ======
        for keyword in KEYWORDS:
            print(f"\n{'='*50}")
            print(f"🚀 开始爬取关键词：{keyword}")
            print(f"{'=' * 50}")

            keyword_jobs = 0

            # ====== 中层循环：页码 ======
            for page_num in range(1, MAX_PAGES_PER_KEYWORD + 1):
                print(f"\n{'-'*50}")
                print(f"📄 开始爬取第 {page_num} 页")
                print(f"{'-' * 50}")

                # 访问列表页
                list_url = LIST_URL_TEMPLATE.format(page=page_num, keyword=keyword)
                print(f"第{page_num}页：{list_url}")
                try:
                    # 加载列表页
                    list_page.goto(list_url, timeout=60000)
                    # 提取详情页URL
                    detail_urls = extract_detail_url(list_page)

                    # 如果本页没数据，说明关键词已到底，提前终止
                    if not detail_urls:
                        print(f"⚠️ 第 {page_num} 页未提取到链接，关键词 '{keyword}' 提前结束")
                        break

                    print(f"✅ 提取到 {len(detail_urls)} 个有效详情页链接")

                    # ====== 内层循环：详情页爬取内容 ======
                    for i, detail_url in enumerate(detail_urls, 1):
                        detail_page = context.new_page()
                        try:
                            print(f"  [{i}/{len(detail_urls)}] 正在抓取: {detail_url}")
                            job = extract_detail(detail_page, detail_url, keyword)
                            if job and job.get('title'):
                                all_jobs.append(job)
                                keyword_jobs += 1
                                print(f"    ✅ 成功: {job['title']} @ {job.get('company', 'N/A')}")
                            else:
                                print(f"    ⚠️ 未提取到有效数据")
                        except Exception as e:
                            print(f"    ❌ 详情页异常：{e}")
                            traceback.print_exc()
                            continue
                        finally:
                            detail_page.close()

                        # 详情页之间随机延迟
                        time.sleep(random.uniform(*DELAY_BETWEEN_DETAILS))
                except Exception as e:
                    print(f"❌ 第 {page_num} 页列表加载失败：{e}")
                    traceback.print_exc()
                    continue

                # 列表页之间随机延迟
                if page_num < MAX_PAGES_PER_KEYWORD:
                    time.sleep(random.uniform(*DELAY_BETWEEN_PAGES))

            print(f"\n🏁 关键词 '{keyword}' 爬取完成，共抓取到 {keyword_jobs} 个岗位")

            # 关键词之间随机延迟
            if keyword != KEYWORDS[-1]:
                delay = random.uniform(*DELAY_BETWEEN_KEYWORDS)
                print(f"⏳ 等待 {delay:.2f} 秒后继续下一个关键词...")
                time.sleep(delay)

        list_page.close()
        browser.close()

    # 保存数据
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"shixiseng_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*50}")
    print(f"✅ 全部完成！总计抓取到 {len(all_jobs)} 个岗位")
    print(f"✅ 数据已保存到文件：{output_file}")
    print(f"\n{'=' * 50}")

if __name__ == "__main__":
    run_spider()
