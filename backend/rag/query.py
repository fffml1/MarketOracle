"""Queries over the existing audited announcement pilot; never ingest data."""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import Lock
from urllib.error import URLError

from .ingest import DIMENSIONS, MODEL, dashscope_embed, load_audited_blocks, metadata, normalize_embedding_endpoint
from .lookup import lookup
from .source import INDEX, read_index, select_version

COVERAGE = {"000858.SZ": ("000858", "wuliangye")}


class RagError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Snapshot:
    output: Path
    digest: str
    version: str
    sources: dict
    blocks: dict
    block_ids: tuple
    positions: dict

    def context(self, identifier):
        block = self.blocks[identifier]
        position = self.positions[identifier]
        section = block.get("section")
        boundary = (block["announcement_id"], block["page_start"], block["page_end"], section)
        excerpt = block["original_text"][:500]
        ids = [identifier]
        for direction in (-1, 1):
            for offset in (1, 2):
                neighbor_at = position + direction * offset
                if not 0 <= neighbor_at < len(self.block_ids):
                    break
                neighbor_id = self.block_ids[neighbor_at]
                neighbor = self.blocks[neighbor_id]
                if (neighbor["kind"] != "正文" or
                    (neighbor["announcement_id"], neighbor["page_start"], neighbor["page_end"], neighbor.get("section")) != boundary):
                    break
                text = neighbor["original_text"]
                preceding = text if direction < 0 else excerpt
                if (len(excerpt) + len(text) + 1 > 500 or preceding.rstrip().endswith(("。", "！", "？", "；"))
                    or direction < 0 and text == section):
                    break
                if direction < 0:
                    ids.insert(0, neighbor_id)
                    excerpt = f"{text}\n{excerpt}"
                else:
                    ids.append(neighbor_id)
                    excerpt = f"{excerpt}\n{text}"
        if len(ids) > 1 and ids[0] == identifier and block["original_text"] == section:
            ids.pop(0)
            excerpt = excerpt.partition("\n")[2]
        return ids, excerpt


def open_collection(path, name):
    from chromadb import PersistentClient
    from chromadb.config import Settings

    client = PersistentClient(path=str(path), settings=Settings(migrations="validate", anonymized_telemetry=False))
    return client.get_collection(name, embedding_function=None)


class AnnouncementQueries:
    def __init__(self, index=INDEX, *, opener=open_collection, embed=dashscope_embed):
        self._index = index
        self._open = opener
        self._embed = embed
        self._snapshots = {}
        self._lock = Lock()

    def _snapshot(self, symbol):
        if symbol not in COVERAGE:
            raise RagError("NOT_COVERED", f"本地财报与公告库尚未覆盖 {symbol}，不代表该公司没有披露")
        stock, directory = COVERAGE[symbol]
        with self._lock:
            if symbol not in self._snapshots:
                output = self._index.parent / "rag" / directory
                try:
                    sources = {key: source for key, source in read_index(self._index).items() if source.code == stock}
                    _, digest, blocks = load_audited_blocks(output)
                    if not blocks or any(
                        (source := sources.get(block["announcement_id"])) is None
                        or (block["stock_code"], block["sha256"], block["disclosed_on"], block["source_url"],
                            block["category"], block["version"])
                        != (stock, source.sha256, source.date, source.url, source.category, source.version)
                        for block in blocks.values()
                    ):
                        raise ValueError("片段与公告索引不符")
                    version = sha256((digest + repr(tuple(sorted(sources.items())))).encode()).hexdigest()
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise RagError("AUDIT_INVALID", "本地公告索引或片段验收无效") from exc
                self._snapshots[symbol] = _Snapshot(output, digest, version, sources, blocks, tuple(blocks),
                                                    {key: index for index, key in enumerate(blocks)})
            return self._snapshots[symbol]

    def version(self, symbol):
        return self._snapshot(symbol).version

    def metric(self, symbol, as_of, period, metric, scope):
        snapshot = self._snapshot(symbol)
        return lookup(snapshot.sources.values(), snapshot.blocks.values(), period, metric, scope, as_of,
                      stock=COVERAGE[symbol][0])

    def _collection(self, snapshot):
        try:
            ready = json.loads((snapshot.output / "chroma_ready.json").read_bytes())
            endpoint = normalize_embedding_endpoint(ready["endpoint"])
            config = {"fragment_file_sha256": snapshot.digest, "model": MODEL,
                      "dimensions": DIMENSIONS, "endpoint": endpoint}
            fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            name = f"{snapshot.output.name}-{fingerprint[:40]}"
            if (ready["complete"] is not True or ready["count"] != len(snapshot.blocks)
                or ready["collection"] != name or any(ready[key] != value for key, value in config.items())):
                raise ValueError("就绪清单与验收快照不符")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RagError("INDEX_INVALID", "公告向量索引未就绪或摘要、模型、维度不符") from exc
        path = snapshot.output / "chroma"
        if not (path / "chroma.sqlite3").is_file():
            raise RagError("COLLECTION_UNAVAILABLE", "本地公告向量集合不存在")
        try:
            collection = self._open(path, name)
            count = collection.count()
        except Exception as exc:
            raise RagError("COLLECTION_UNAVAILABLE", "无法打开本地公告向量集合") from exc
        if collection.metadata != config or count != len(snapshot.blocks):
            raise RagError("INDEX_INVALID", "公告向量集合配置不符或片段不完整")
        return collection, endpoint

    def search(self, symbol, as_of, query, period=None, category=None):
        snapshot = self._snapshot(symbol)
        stock = COVERAGE[symbol][0]
        candidates = set()
        reports = set()
        for source in snapshot.sources.values():
            if source.date > as_of or (category and source.category != category) or (period and source.period != period):
                continue
            if source.category == "定期报告" and source.period:
                reports.add(source.period)
            else:
                candidates.add(source.id)
        for report_period in reports:
            source = select_version(snapshot.sources.values(), stock, report_period, "定期报告", as_of)
            candidates.add(source.id)
        if not candidates:
            return []

        collection, recorded_endpoint = self._collection(snapshot)
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise RagError("EMBEDDING_CONFIG", "未配置公告检索的向量密钥")
        try:
            endpoint = normalize_embedding_endpoint(os.getenv("MARKETORACLE_EMBEDDING_URL", ""))
            if endpoint != recorded_endpoint:
                raise ValueError("向量服务地址与就绪清单不符")
        except ValueError as exc:
            raise RagError("EMBEDDING_CONFIG", "公告检索向量地址无效或与入库地址不一致") from exc
        try:
            vector, = self._embed(endpoint, key, [query], text_type="query", timeout=10)
        except (TimeoutError, URLError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(exc.reason, TimeoutError):
                raise RagError("EMBEDDING_TIMEOUT", "公告检索查询向量请求超时") from exc
            raise RagError("EMBEDDING_FAILED", "公告检索查询向量服务不可用") from exc
        except Exception as exc:
            raise RagError("EMBEDDING_FAILED", "公告检索查询向量服务不可用") from exc
        try:
            valid_vector = len(vector) == DIMENSIONS and all(math.isfinite(value) for value in vector)
        except (TypeError, ValueError, OverflowError):
            valid_vector = False
        if not valid_vector:
            raise RagError("EMBEDDING_INVALID", "公告检索查询向量维度或数值无效")

        where = {"$and": [{"stock_code": stock}, {"announcement_id": {"$in": sorted(candidates)}},
                          {"disclosed_on": {"$lte": int(as_of.replace("-", ""))}}, {"kind": "正文"}]}
        try:
            result = collection.query(query_embeddings=[vector], where=where, n_results=3, include=["metadatas"])
        except Exception as exc:
            raise RagError("SEARCH_FAILED", "本地公告向量检索失败") from exc
        try:
            hits = {}
            for identifier, meta in zip(result["ids"][0], result["metadatas"][0], strict=True):
                block = snapshot.blocks[identifier]
                if (block["announcement_id"] not in candidates or block["kind"] != "正文"
                    or block["disclosed_on"] > as_of or meta != metadata(block)):
                    raise ValueError("命中片段未通过校验")
                hits[identifier] = block
            excerpts = []
            cited = set()
            for identifier, block in list(hits.items())[:3]:
                if identifier in cited:
                    continue
                fragment_ids, excerpt = snapshot.context(identifier)
                if all(fragment_id in cited for fragment_id in fragment_ids):
                    continue
                cited.update(fragment_ids)
                excerpts.append({"fragment_id": identifier, "fragment_ids": fragment_ids,
                                 "announcement_id": block["announcement_id"],
                                 "title": snapshot.sources[block["announcement_id"]].title,
                                 "version": block["version"], "category": block["category"],
                                 "disclosed_on": block["disclosed_on"], "page_start": block["page_start"],
                                 "page_end": block["page_end"], "source_url": block["source_url"], "excerpt": excerpt})
            return excerpts
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise RagError("HIT_INVALID", "检索命中与已验收公告片段不一致") from exc


queries = AnnouncementQueries()
