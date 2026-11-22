"""Resumable Chroma ingestion for the audited Wuliangye pilot."""
import argparse
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .source import OUTPUT, STOCK

MODEL = "qwen3.7-text-embedding-flash"
DIMENSIONS = 1024
BATCH_SIZE = 10


def normalize_embedding_endpoint(endpoint):
    endpoint = endpoint.strip().rstrip("/")
    if endpoint.endswith("/compatible-mode/v1"):
        endpoint += "/embeddings"
    url = urlsplit(endpoint)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
        or url.query or url.fragment
        or not url.path.endswith(("/text-embedding/text-embedding", "/compatible-mode/v1/embeddings"))):
        raise ValueError("百炼文本向量地址无效")
    return endpoint


def dashscope_embed(endpoint, key, texts, *, text_type="document", timeout=60):
    compatible = endpoint.endswith("/compatible-mode/v1/embeddings")
    if compatible:
        payload = {"model": MODEL, "input": texts, "dimensions": DIMENSIONS}
    else:
        payload = {"model": MODEL, "input": {"texts": texts},
                   "parameters": {"dimension": DIMENSIONS, "text_type": text_type}}
    request = Request(endpoint, data=json.dumps(payload).encode(),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            output = data["data"] if compatible else data["output"]["embeddings"]
    except HTTPError as error:
        raise RuntimeError(f"百炼向量接口返回 HTTP {error.code}，请核对地域、权限和模型可用性") from None
    vectors = {item.get("index", item.get("text_index")): item["embedding"] for item in output}
    if len(output) != len(texts) or set(vectors) != set(range(len(texts))):
        raise ValueError("向量接口返回的输入序号不匹配")
    return [vectors[index] for index in range(len(texts))]


def metadata(block):
    keys = ("stock_code", "announcement_id", "period", "category", "version", "kind",
            "page_start", "page_end", "sha256", "source_url", "parser", "rule_version")
    return {**{key: block[key] for key in keys},
            "disclosed_on": int(block["disclosed_on"].replace("-", ""))}


def load_audited_blocks(output):
    report = json.loads((output / "report.json").read_bytes())
    data = (output / "fragments.jsonl").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not report["passed"] or digest != report["fragment_file_sha256"]:
        raise ValueError("解析尚未通过验收，或片段文件与验收报告不一致")
    blocks = {block["id"]: block for line in data.splitlines() if (block := json.loads(line))}
    if len(blocks) != report["source_checks"]["fragment_citations_checked"]:
        raise ValueError("片段 ID 数量与报告不符")
    return report, digest, blocks


def ingest(collection, blocks, embed, max_batches=None):
    written = 0
    reused = 0
    batches_written = 0
    ids_in_order = list(blocks)
    for offset in range(0, len(ids_in_order), BATCH_SIZE):
        ids = ids_in_order[offset:offset + BATCH_SIZE]
        present = collection.get(ids=ids, include=["documents", "metadatas"])
        for identifier, document, meta in zip(present["ids"], present["documents"], present["metadatas"],
                                              strict=True):
            if document != blocks[identifier]["text"] or meta != metadata(blocks[identifier]):
                raise ValueError(f"集合中的既有片段与待入库内容不符：{identifier}")
        known = set(present["ids"])
        pending = [identifier for identifier in ids if identifier not in known]
        reused += len(known)
        if not pending:
            continue
        if max_batches is not None and batches_written >= max_batches:
            return {"complete": False, "written": written, "reused": reused}
        documents = [blocks[identifier]["text"] for identifier in pending]
        vectors = embed(documents)
        if len(vectors) != len(pending) or any(
            len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector) for vector in vectors
        ):
            raise ValueError("向量数量、维度或数值无效")
        collection.upsert(ids=pending, documents=documents, embeddings=vectors,
                          metadatas=[metadata(blocks[identifier]) for identifier in pending])
        batches_written += 1
        written += len(pending)
    if collection.count() != len(blocks):
        raise ValueError("集合数量与通过校验的片段 ID 数量不一致")
    return {"complete": True, "written": written, "reused": reused}


def verify_versions(collection, blocks, checks):
    for check in checks:
        cutoff = int(date.fromisoformat(check["as_of"]).strftime("%Y%m%d"))
        expected_id = check["selected_id"]
        expected = {identifier for identifier, block in blocks.items()
                    if block["announcement_id"] == expected_id and block["kind"] == "表格行"}
        where = {"$and": [{"stock_code": STOCK}, {"announcement_id": expected_id},
                          {"disclosed_on": {"$lte": cutoff}}, {"kind": "表格行"}]}
        found = collection.get(where=where, include=[])["ids"]
        if not expected or set(found) != expected:
            raise ValueError(f"截止日 {check['as_of']} 的公告版本检索验收失败")


def main():
    from dotenv import load_dotenv
    from .validate import write_records

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-batches", type=int, help="写入指定批数后退出，供断点恢复验收")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches 必须为正整数")
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    try:
        endpoint = normalize_embedding_endpoint(os.getenv("MARKETORACLE_EMBEDDING_URL", ""))
        if not key:
            raise ValueError("缺少密钥")
    except ValueError:
        parser.error("请配置 DASHSCOPE_API_KEY 和百炼文本向量 HTTPS 地址 MARKETORACLE_EMBEDDING_URL")
    try:
        import chromadb
    except ModuleNotFoundError as error:
        if error.name != "chromadb":
            raise
        parser.error("当前 Python 环境缺少 chromadb")
    report, digest, blocks = load_audited_blocks(args.output)
    config = {"fragment_file_sha256": digest, "model": MODEL, "dimensions": DIMENSIONS,
              "endpoint": endpoint}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    client = chromadb.PersistentClient(path=str(args.output / "chroma"))
    collection = client.get_or_create_collection(name=f"wuliangye-{fingerprint[:40]}",
                                                 embedding_function=None, metadata=config)
    if collection.metadata != config:
        raise ValueError("集合配置与本次入库不符")
    result = ingest(collection, blocks, lambda texts: dashscope_embed(endpoint, key, texts), args.max_batches)
    print(json.dumps(result, ensure_ascii=False))
    if result["complete"]:
        verify_versions(collection, blocks, report["version_checks"])
        write_records(args.output / "chroma_ready.json",
                      {**config, **result, "collection": collection.name, "count": collection.count()},
                      jsonl=False)


if __name__ == "__main__":
    main()
