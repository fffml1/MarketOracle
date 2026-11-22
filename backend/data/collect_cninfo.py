from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import csv, hashlib, json, re, time

ROOT = Path(__file__).resolve().parent
START, END = "2021-11-22", "2026-11-22"
RULES = {
    "定期报告": r"20\d\d年?(年度|半年度|第?[一三]季度)报告(全文)?([（(].*?[）)])?$",
    "补充更正": "更正|会计差错|追溯调整",
    "业绩预告与快报": "业绩预告|业绩快报|初步核算|主要财务数据",
    "分红": "权益分派|利润分配|现金分红|特别分红|分红派息",
    "经营与重大投资": "主要经营数据|经营情况|重大合同|重大投资|投资建设|签[署订].*(协议|合同)",
    "回购与增减持": "回购.*(方案|结果)|[增减]持.*(计划|实施结果|实施完成)",
    "风险与监管": "风险提示|重大诉讼|重大仲裁|行政处罚|监管函|问询函|立案调查|澄清公告",
    "投资者交流": "投资者关系活动记录|业绩说明会.*(情况|记录|纪要)",
}

def fetch(url, form=None):
    for attempt in range(4):
        try:
            req = Request(url, data=urlencode(form).encode() if form else None, headers={"User-Agent":"Mozilla/5.0", "Referer":"https://www.cninfo.com.cn/"})
            with urlopen(req, timeout=55) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)

def inventory(job):
    code, name, org, year = job
    start, end = max(START, f"{year}-01-01"), min(END, f"{year}-12-31")
    found = []
    for page in range(1, 100):
        form = dict(stock=f"{code},{org}", column="sse" if code=="600519" else "szse", tabName="fulltext", pageNum=page, pageSize=30, seDate=f"{start}~{end}")
        data = json.loads(fetch("https://www.cninfo.com.cn/new/hisAnnouncement/query", form))
        for item in data["announcements"] or []:
            assert item["secCode"] == code
            title = re.sub("<[^>]+>", "", item["announcementTitle"]).strip()
            if re.search("摘要|英文|English|法律意见|核查意见|独立意见|保荐|独立财务顾问", title, re.I):
                continue
            category = next((k for k,v in RULES.items() if re.search(v,title)), None)
            path = item["adjunctUrl"] or ""
            if not category or not re.fullmatch(r"finalpage/\d{4}-\d\d-\d\d/\d+\.pdf",path,re.I):
                continue
            date = path.split("/")[1]
            if not START <= date <= END:
                continue
            aid = item["announcementId"]
            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)[:85]
            local = Path(name)/date[:4]/category/f"{date}_{safe}_{aid}.pdf"
            found.append(dict(code=code,company=name,category=category,date=date,title=title,id=aid,url="https://static.cninfo.com.cn/"+path,file=local.as_posix()))
        if page * 30 >= data["totalAnnouncement"]:
            break
    print(name, year, len(found), flush=True)
    return found

def download(row):
    path = ROOT / row["file"]
    blob = path.read_bytes() if path.exists() else fetch(row["url"])
    if not blob.startswith(b"%PDF-") or b"%%EOF" not in blob[-4096:]:
        raise ValueError(row["url"])
    if not path.exists():
        path.parent.mkdir(exist_ok=True,parents=True)
        part = path.with_suffix(".part")
        part.write_bytes(blob)
        part.replace(path)
    return dict(row,sha256=hashlib.sha256(blob).hexdigest())

if __name__ == "__main__":
    stocks = json.loads(fetch("https://www.cninfo.com.cn/new/data/szse_stock.json"))["stockList"]
    jobs = []
    for code,name in [("600519","贵州茅台"),("000858","五粮液"),("300750","宁德时代")]:
        org = "gssh0600519" if code=="600519" else next(s["orgId"] for s in stocks if s["code"]==code)
        jobs.extend((code,name,org,y) for y in range(2021,2027))
    with ThreadPoolExecutor(3) as pool:
        rows = {r["id"]:r for group in pool.map(inventory,jobs) for r in group}
    print("Selected", len(rows), flush=True)
    with ThreadPoolExecutor(4) as pool:
        done = list(pool.map(download,rows.values()))
    done.sort(key=lambda r:(r["code"],r["date"],r["id"]))
    with (ROOT/"公告索引.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=done[0].keys())
        writer.writeheader()
        writer.writerows(done)


