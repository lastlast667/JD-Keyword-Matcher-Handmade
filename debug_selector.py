"""
调试脚本：确认实习僧列表页的实际HTML结构
运行这个，看控制台输出，找到正确的选择器
"""

from playwright.sync_api import sync_playwright

URL = "https://www.shixiseng.com/interns?page=1&keyword=java&type=intern&city=全国"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False) # 非headless模式，方便调试
    page = browser.new_page()   # 打开一个新页面
    page.goto(URL,timeout=30000,wait_until="domcontentloaded")    # 访问URL

    # 等待页面加载完成
    page.wait_for_selector(".result-list.clearfix")

    print("="*50)
    print("测试1：找包含 /intern/ 链接的 a 标签")
    print("="*50)

    # 先找到所有列表项的a标签，再筛选
    list_items = page.query_selector_all(".intern-wrap.interns-point.intern-item a")
    print(f"找到 {len(list_items)} 个列表项")

    intern_links = []
    for item in list_items:
        href = item.get_attribute("href")
        if href and "/intern/" in href:
            intern_links.append(href)

    print(f"找到 {len(intern_links)} 个包含 /intern/ 链接的 a 标签")
    if intern_links:
        print(f"第一个链接：{intern_links[0]}")

    
