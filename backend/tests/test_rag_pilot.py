import unittest

try:
    import pdfplumber
except ModuleNotFoundError as error:
    raise unittest.SkipTest("离线 PDF 验收需要 pdfplumber") from error

from backend.rag.ingest import BATCH_SIZE, DIMENSIONS, ingest
from backend.rag.lookup import lookup
from backend.rag.source import Announcement


class StoredCollection:
    def __init__(self):
        self.records = {}

    def get(self, ids, include):
        records = [(identifier, *self.records[identifier]) for identifier in ids if identifier in self.records]
        return {key: [record[position] for record in records]
                for position, key in enumerate(("ids", "documents", "metadatas"))}

    def upsert(self, ids, documents, embeddings, metadatas):
        self.records.update(zip(ids, zip(documents, metadatas, strict=True), strict=True))

    def count(self):
        return len(self.records)


class PilotTests(unittest.TestCase):
    def test_interrupted_ingestion_resumes_without_reembedding_or_duplicate_ids(self):
        collection = StoredCollection()
        block = {"stock_code": "000858", "announcement_id": "a", "period": "2025年三季度",
                 "category": "定期报告", "version": "披露版", "kind": "表格行",
                 "page_start": 1, "page_end": 1, "sha256": "abc", "source_url": "https://example.org/a.pdf",
                 "parser": "pdfplumber", "rule_version": "1", "disclosed_on": "2025-10-31"}
        blocks = {str(number): {**block, "text": str(number)} for number in range(BATCH_SIZE + 1)}
        embedded = []

        def embed(texts):
            embedded.extend(texts)
            return [[1.0] * DIMENSIONS for _ in texts]

        partial = ingest(collection, blocks, embed, max_batches=1)
        self.assertFalse(partial["complete"])
        self.assertEqual(collection.count(), BATCH_SIZE)
        finished = ingest(collection, blocks, embed)
        self.assertEqual(finished, {"complete": True, "written": 1, "reused": BATCH_SIZE})
        repeated = ingest(collection, blocks, embed)
        self.assertEqual(repeated, {"complete": True, "written": 0, "reused": len(blocks)})
        self.assertEqual(embedded, list(blocks))

    def test_lookup_keeps_single_period_and_does_not_fall_back_to_old_version(self):
        before = Announcement("a", "000858", "五粮液", "定期报告", "2025-10-31",
                              "2025年三季度报告（更新前）", "https://example.org/a.pdf", "a.pdf", "a")
        after = Announcement("b", "000858", "五粮液", "定期报告", "2026-04-30",
                             "2025年三季度报告（更新后）", "https://example.org/b.pdf", "b.pdf", "b")
        fields = [{"role": "数值", "unit": "元", "header": {"text": header, "page": 1},
                   "value": {"text": value, "page": 1}, "unit_source": {"page": 1}}
                  for header, value in (("本报告期", "100"), ("年初至报告期末", "250"))]
        blocks = [{"announcement_id": "a", "sha256": "a", "kind": "表格行", "section": "主要会计数据",
                   "metric": {"text": "营业收入（元）"}, "fields": fields}]
        selected = lookup([before, after], blocks, before.period, "营业收入", "本期", "2025-11-01")
        self.assertEqual(selected["announcement_id"], "a")
        self.assertEqual([row["value"] for row in selected["evidence"]], ["100"])
        selected = lookup([before, after], blocks, before.period, "营业收入", "本期", "2026-05-01")
        self.assertEqual(selected["announcement_id"], "b")
        self.assertEqual(selected["status"], "所选版本缺少通过校验的数字")
        self.assertEqual(selected["evidence"], [])


if __name__ == "__main__":
    unittest.main()
