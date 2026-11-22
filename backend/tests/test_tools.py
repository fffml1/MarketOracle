from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from backend.schemas.evidence import EvidenceCategory
from backend.tools.base_evidence_tool import BaseEvidenceTool
from backend.tools.market_data import MarketTechnicalTool
from backend.tools.news_catalyst import NewsCatalystTool
from backend.tools.risk_notice import RiskAndNoticeTool


class BaseEvidenceToolTests(unittest.TestCase):
    def test_next_evidence_id_and_default_category(self) -> None:
        tool = BaseEvidenceTool()
        tool.id_prefix = "EV_TEST"

        self.assertEqual(tool.next_evidence_id(), "EV_TEST_001")
        item = tool.build_evidence(
            symbol="600519.SH",
            title="测试证据",
            summary="摘要",
            source="unit_test",
        )

        self.assertEqual(item.evidence_id, "EV_TEST_002")
        self.assertEqual(item.category, EvidenceCategory.TECHNICAL)

    def test_normalize_as_of_date(self) -> None:
        self.assertEqual(
            BaseEvidenceTool.normalize_as_of_date("2026-11-21"),
            "2026-11-21",
        )
        self.assertEqual(
            BaseEvidenceTool.normalize_as_of_date(None),
            date.today().isoformat(),
        )
        with self.assertRaises(ValueError):
            BaseEvidenceTool.normalize_as_of_date("2026/11/21")

    def test_bundle_is_partial_when_item_is_degraded(self) -> None:
        tool = BaseEvidenceTool()
        item = tool.build_evidence(
            symbol="600519.SH",
            title="降级证据",
            summary="摘要",
            source="unit_test",
            is_degraded=True,
            degrade_reason="数据不完整",
        )
        bundle = tool.build_bundle(
            symbol="600519.SH",
            as_of_date="2026-11-21",
            evidences=[item],
        )

        self.assertEqual(bundle.status, "partial")
        self.assertTrue(bundle.is_degraded)


class MarketTechnicalToolTests(unittest.TestCase):
    @staticmethod
    def make_kline(rows: int = 25) -> pd.DataFrame:
        end = pd.Timestamp("2026-11-21")
        records = []
        for index in range(rows):
            trade_date = end - pd.Timedelta(days=rows - 1 - index)
            close = 100.0 + index
            records.append(
                {
                    "date": trade_date.strftime("%Y-%m-%d"),
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "volume": 1000 + index * 10,
                }
            )
        return pd.DataFrame(records)

    def test_default_fetcher_switches_to_tencent_on_disconnect(self) -> None:
        kline = self.make_kline()
        kline["date"] = pd.to_datetime(kline["date"]).dt.date
        with (
            patch("akshare.stock_zh_a_hist_em", side_effect=ConnectionError("disconnected")) as eastmoney,
            patch("akshare.stock_zh_a_hist_tx", return_value=kline) as tencent,
        ):
            bundle = MarketTechnicalTool().execute("300750.SZ", "2026-11-21")

        eastmoney.assert_called_once()
        tencent.assert_called_once_with(
            symbol="sz300750", start_date="20260525",
            end_date="20261121", adjust="qfq",
        )
        self.assertEqual(bundle.status, "ok")
        item = bundle.evidences[0]
        self.assertEqual(item.source, "akshare_tencent_qfq")
        self.assertEqual(item.payload["volume_unit"], "股")
        self.assertEqual(item.payload["close"], 124.0)

    def test_default_fetcher_keeps_eastmoney_when_available(self) -> None:
        kline = self.make_kline().rename(columns={
            "date": "日期", "open": "开盘", "close": "收盘",
            "high": "最高", "low": "最低", "volume": "成交量",
        })
        with (
            patch("akshare.stock_zh_a_hist_em", return_value=kline),
            patch("akshare.stock_zh_a_hist_tx") as tencent,
        ):
            bundle = MarketTechnicalTool().execute("300750.SZ", "2026-11-21")

        tencent.assert_not_called()
        self.assertEqual(bundle.evidences[0].source, "akshare_eastmoney_qfq")
        self.assertEqual(bundle.evidences[0].payload["volume_unit"], "手")

    def test_default_fetcher_reports_both_failures(self) -> None:
        with (
            patch("akshare.stock_zh_a_hist_em", return_value=pd.DataFrame()),
            patch("akshare.stock_zh_a_hist_tx", side_effect=TimeoutError("timed out")),
        ):
            bundle = MarketTechnicalTool().execute("300750.SZ", "2026-11-21")

        self.assertEqual(bundle.status, "error")
        self.assertIn("akshare_eastmoney_qfq", bundle.error_message or "")
        self.assertIn("akshare_tencent_qfq", bundle.error_message or "")

    def test_execute_returns_metrics_and_filters_future_rows(self) -> None:
        kline = self.make_kline(25)
        kline = pd.concat(
            [
                kline,
                pd.DataFrame(
                    [
                        {
                            "date": "2026-11-22",
                            "open": 999,
                            "high": 999,
                            "low": 999,
                            "close": 999,
                            "volume": 999,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        tool = MarketTechnicalTool(kline_fetcher=lambda *_args: kline)

        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "ok")
        self.assertFalse(bundle.is_degraded)
        self.assertEqual(bundle.evidences[0].evidence_id, "EV_TECH_001")
        self.assertEqual(bundle.evidences[0].published_at, "2026-11-21")
        self.assertEqual(bundle.evidences[0].payload["bars_count"], 25)
        self.assertEqual(bundle.evidences[0].payload["close"], 124.0)

    def test_execute_marks_short_history_as_degraded(self) -> None:
        tool = MarketTechnicalTool(kline_fetcher=lambda *_args: self.make_kline(10))

        bundle = tool.execute("300750.SZ", "2026-11-21")

        self.assertEqual(bundle.status, "partial")
        self.assertTrue(bundle.is_degraded)
        self.assertIn("少于20日", bundle.degrade_reason or "")
        self.assertTrue(bundle.evidences[0].is_degraded)

    def test_execute_returns_error_when_fetcher_raises(self) -> None:
        def failing_fetcher(*_args: object) -> pd.DataFrame:
            raise RuntimeError("market api unavailable")

        tool = MarketTechnicalTool(kline_fetcher=failing_fetcher)
        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "error")
        self.assertEqual(bundle.error_message, "market api unavailable")
        self.assertEqual(bundle.evidences, [])

    def test_execute_returns_error_when_no_data_before_anchor(self) -> None:
        tool = MarketTechnicalTool(
            kline_fetcher=lambda *_args: pd.DataFrame(
                [{"date": "2026-11-22", "close": 100}]
            )
        )
        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "error")
        self.assertIn("无可用交易日", bundle.degrade_reason or "")


class NewsCatalystToolTests(unittest.TestCase):
    def test_execute_filters_window_deduplicates_and_truncates(self) -> None:
        rows = [
            {
                "title": "业绩增长",
                "summary": "N" * 250,
                "published_at": "2026-11-20",
                "source": "eastmoney",
            },
            {
                "title": "业绩增长",
                "summary": "重复新闻",
                "published_at": "2026-11-19",
                "source": "eastmoney",
            },
            {
                "title": "未来新闻",
                "summary": "不应出现",
                "published_at": "2026-11-22",
                "source": "eastmoney",
            },
            {
                "title": "过期新闻",
                "summary": "不应出现",
                "published_at": "2026-10-01",
                "source": "eastmoney",
            },
        ]
        tool = NewsCatalystTool(news_fetcher=lambda *_args: rows, summary_max_len=20)

        bundle = tool.execute("600519.SH", "2026-11-21", lookback_days=14)

        self.assertEqual(bundle.status, "ok")
        self.assertEqual(len(bundle.evidences), 1)
        self.assertEqual(bundle.evidences[0].title, "业绩增长")
        self.assertTrue(bundle.evidences[0].summary.endswith("..."))
        self.assertEqual(bundle.evidences[0].source, "eastmoney")

    def test_execute_marks_missing_dates_as_degraded(self) -> None:
        tool = NewsCatalystTool(
            news_fetcher=lambda *_args: [
                {"title": "日期未知", "summary": "待核实", "published_at": None}
            ]
        )

        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "partial")
        self.assertTrue(bundle.is_degraded)
        self.assertTrue(bundle.evidences[0].is_degraded)
        self.assertIn("缺少明确发布日期", bundle.degrade_reason or "")

    def test_execute_returns_partial_when_window_has_no_matches(self) -> None:
        tool = NewsCatalystTool(
            news_fetcher=lambda *_args: [
                {"title": "未来新闻", "published_at": "2026-12-01"}
            ]
        )

        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "partial")
        self.assertIn("时间窗口内无匹配新闻", bundle.degrade_reason or "")

    def test_execute_returns_error_when_fetcher_raises(self) -> None:
        def failing_fetcher(*_args: object) -> list[dict[str, object]]:
            raise RuntimeError("news api unavailable")

        tool = NewsCatalystTool(news_fetcher=failing_fetcher)
        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "error")
        self.assertEqual(bundle.error_message, "news api unavailable")


class RiskAndNoticeToolTests(unittest.TestCase):
    def test_execute_adds_risk_tags_and_limits_items(self) -> None:
        rows = [
            {
                "title": "股东计划减持",
                "summary": "因自身资金需要",
                "published_at": "2026-11-20",
                "source": "公告",
            },
            {
                "title": "股东计划减持",
                "summary": "重复公告",
                "published_at": "2026-11-19",
                "source": "公告",
            },
        ]
        tool = RiskAndNoticeTool(risk_fetcher=lambda *_args: rows, max_items=8)

        bundle = tool.execute("600519.SH", "2026-11-21", lookback_days=30)

        self.assertEqual(bundle.status, "ok")
        self.assertEqual(len(bundle.evidences), 1)
        self.assertIn("减持", bundle.evidences[0].payload["risk_tags"])
        self.assertIn("标签: 减持", bundle.evidences[0].summary)

    def test_execute_marks_missing_date_as_degraded(self) -> None:
        tool = RiskAndNoticeTool(
            risk_fetcher=lambda *_args: [
                {
                    "title": "公司收到监管问询",
                    "summary": "需要说明事项",
                    "published_at": None,
                    "risk_tags": ["问询"],
                }
            ]
        )

        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "partial")
        self.assertTrue(bundle.is_degraded)
        self.assertTrue(bundle.evidences[0].is_degraded)
        self.assertEqual(bundle.evidences[0].published_at, None)

    def test_execute_returns_partial_when_no_rows(self) -> None:
        tool = RiskAndNoticeTool(risk_fetcher=lambda *_args: [])

        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "partial")
        self.assertTrue(bundle.is_degraded)
        self.assertIn("未检出高危公告", bundle.degrade_reason or "")

    def test_execute_returns_error_when_fetcher_raises(self) -> None:
        def failing_fetcher(*_args: object) -> list[dict[str, object]]:
            raise RuntimeError("risk api unavailable")

        tool = RiskAndNoticeTool(risk_fetcher=failing_fetcher)
        bundle = tool.execute("600519.SH", "2026-11-21")

        self.assertEqual(bundle.status, "error")
        self.assertEqual(bundle.error_message, "risk api unavailable")


if __name__ == "__main__":
    unittest.main()
