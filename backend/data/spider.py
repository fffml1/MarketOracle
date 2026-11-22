import akshare as ak
import pandas as pd
import os
import time
import random
import requests

# ========== 配置区 ==========
STOCK_CODES = ["000858", "600519"]
START_DATE = "20210101"
END_DATE = "20261122"
SAVE_CSV_PATH = "cninfo_notice_list.csv"
DOWNLOAD_PDF = False
PDF_SAVE_DIR = "./cninfo_pdf"
KEY_FILTER = ["年度报告", "半年度报告", "季度报告", "投资者关系活动", "权益分派", "分红"]
# ============================

os.makedirs(PDF_SAVE_DIR, exist_ok=True)
all_data = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"
}

def download_pdf(url, code, title):
    safe_title = "".join([c for c in title if c.isalnum() or c in (" ", "_")]).strip()
    save_name = f"{code}_{safe_title}.pdf"
    save_path = os.path.join(PDF_SAVE_DIR, save_name)
    if os.path.exists(save_path):
        return
    try:
        r = requests.get(url, headers=headers, timeout=20)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ 下载 {save_name}")
        time.sleep(random.uniform(2, 3))
    except Exception as e:
        print(f"❌ 下载失败 {url} {e}")

for sc in STOCK_CODES:
    print(f"\n===== 抓取 {sc} =====")
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=sc,
            start_date=START_DATE,
            end_date=END_DATE
        )
        print(f"【{sc}】当前返回列名：{df.columns.tolist()}")
    except Exception as e:
        print(f"【{sc}】接口报错：{e}")
        continue

    def match_title(t):
        return any(k in t for k in KEY_FILTER)
    df = df[df["公告标题"].apply(match_title)]

    for _, row in df.iterrows():
        pdf_url = row["公告链接"]
        item = {
            "stock_code": sc,
            "title": row["公告标题"],
            "publish_time": row["公告时间"],
            "pdf_url": pdf_url,
            "source": "巨潮资讯 cninfo"
        }
        all_data.append(item)
        if DOWNLOAD_PDF:
            download_pdf(pdf_url, sc, row["公告标题"])
    time.sleep(random.uniform(3,5))

df_out = pd.DataFrame(all_data)
df_out.to_csv(SAVE_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\n✅ 完成，一共 {len(df_out)} 条，保存到 {SAVE_CSV_PATH}")
