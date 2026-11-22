from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..schemas.evidence import EvidenceBundle, EvidenceCategory, EvidenceItem
from .base_evidence_tool import BaseEvidenceTool

# 数据源函数类型契约：输入 (symbol, as_of_date, lookback_days)，返回包含 date, open, high, low, close, volume 的 DataFrame
KlineFetcher = Callable[[str, str, int], pd.DataFrame]


def default_akshare_kline_fetcher(symbol: str, as_of_date: str, lookback_days: int = 180) -> pd.DataFrame:
    """
    默认 A 股日线获取器（基于 akshare）。
    支持传入如 '300750.SZ'、'600519.SH' 或纯 6 位代码 '300750'。
    """
    import akshare as ak  # 延迟导入，避免未装 akshare 时影响纯单元测试

    code, _, exchange = symbol.strip().upper().partition(".")
    start_date = (pd.to_datetime(as_of_date) - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = pd.to_datetime(as_of_date).strftime("%Y%m%d")
    date_args = {"start_date": start_date, "end_date": end_date, "adjust": "qfq"}
    tx_symbol = f"{exchange.lower()}{code}" if exchange in {"SH", "SZ", "BJ"} else code
    sources = (
        ("akshare_eastmoney_qfq", "手", lambda: ak.stock_zh_a_hist_em(
            symbol=code, period="daily", **date_args
        )),
        ("akshare_tencent_qfq", "股", lambda: ak.stock_zh_a_hist_tx(
            symbol=tx_symbol, **date_args
        )),
    )
    failures = []
    for source, volume_unit, fetch in sources:
        try:
            df = _normalize_kline(fetch(), start_date, end_date)
            if not df.empty:
                df.attrs.update(source=source, volume_unit=volume_unit)
                return df
            failures.append(f"{source}：请求区间内无有效日线")
        except Exception as exc:
            failures.append(f"{source}：{exc}")
    raise RuntimeError("；".join(failures))


def _normalize_kline(df: pd.DataFrame | None, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
    }
    df = df.rename(columns=rename_map)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in required - {"date"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return (
        df.dropna(subset=sorted(required))
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )


class MarketTechnicalTool(BaseEvidenceTool):
    """行情与量价技术指标工具（纯 Python 计算，输出带 evidence_id 的 EvidenceBundle）"""

    tool_name: str = "get_market_technical"
    id_prefix: str = "EV_TECH"
    default_category: EvidenceCategory = EvidenceCategory.TECHNICAL
    DEFAULT_LOOKBACK_DAYS = 180

    def __init__(
        self,
        kline_fetcher: KlineFetcher | None = None,
        source_name: str = "akshare_hist_qfq",
    ) -> None:
        super().__init__()
        self._kline_fetcher: KlineFetcher = kline_fetcher or default_akshare_kline_fetcher
        self._source_name = source_name

    @staticmethod
    def _compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
        """用确定性代码计算均线、涨跌幅、回撤与量能状态"""
        latest = df.iloc[-1]
        close_now = float(latest["close"])
        latest_date = str(latest["date"])

        # 区间涨跌幅（按收盘价计算）
        ret_5d = (
            float((close_now / df.iloc[-6]["close"] - 1.0) * 100)
            if len(df) >= 6 and float(df.iloc[-6]["close"]) != 0
            else None
        )
        ret_20d = (
            float((close_now / df.iloc[-21]["close"] - 1.0) * 100)
            if len(df) >= 21 and float(df.iloc[-21]["close"]) != 0
            else None
        )

        # 均线计算
        ma5 = float(df["close"].tail(5).mean()) if len(df) >= 5 else None
        ma20 = float(df["close"].tail(20).mean()) if len(df) >= 20 else None
        ma60 = float(df["close"].tail(60).mean()) if len(df) >= 60 else None

        # 均线形态判断
        if ma5 is not None and ma20 is not None and ma60 is not None:
            if close_now > ma5 > ma20 > ma60:
                ma_alignment = "多头排列（价格 > MA5 > MA20 > MA60）"
            elif close_now < ma5 < ma20 < ma60:
                ma_alignment = "空头排列（价格 < MA5 < MA20 < MA60）"
            else:
                ma_alignment = "均线交织/震荡整理"
        else:
            ma_alignment = "历史交易日不足60天，均线不完整"

        # 近 60 个交易日最大回撤（基于收盘价）
        win_60 = df.tail(60)
        rolling_peak = win_60["close"].cummax()
        drawdown_series = (win_60["close"] - rolling_peak) / rolling_peak * 100
        max_dd_60d = float(drawdown_series.min()) if not drawdown_series.empty else None

        # 近 60 日区间高低点
        high_60d = float(win_60["high"].max()) if not win_60.empty else None
        low_60d = float(win_60["low"].min()) if not win_60.empty else None

        # 量能变化：最新成交量 vs 近 5 日均量 / 近 20 日均量
        vol_now = float(latest["volume"])
        vol_ma5 = float(df["volume"].tail(5).mean()) if len(df) >= 5 else None
        vol_ma20 = float(df["volume"].tail(20).mean()) if len(df) >= 20 else None
        vol_ratio_20d = float(vol_now / vol_ma20) if vol_ma20 and vol_ma20 > 0 else None

        return {
            "latest_trade_date": latest_date,
            "bars_count": int(len(df)),
            "close": round(close_now, 4),
            "close_series": [
                {"date": str(day), "close": round(float(close), 4)}
                for day, close in df[["date", "close"]].tail(20).itertuples(index=False, name=None)
            ],
            "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
            "ret_20d_pct": round(ret_20d, 2) if ret_20d is not None else None,
            "ma5": round(ma5, 4) if ma5 is not None else None,
            "ma20": round(ma20, 4) if ma20 is not None else None,
            "ma60": round(ma60, 4) if ma60 is not None else None,
            "ma_alignment": ma_alignment,
            "max_drawdown_60d_pct": round(max_dd_60d, 2) if max_dd_60d is not None else None,
            "high_60d": round(high_60d, 4) if high_60d is not None else None,
            "low_60d": round(low_60d, 4) if low_60d is not None else None,
            "vol_ratio_20d": round(vol_ratio_20d, 2) if vol_ratio_20d is not None else None,
            "vol_ma5": round(vol_ma5, 2) if vol_ma5 is not None else None,
            "vol_ma20": round(vol_ma20, 2) if vol_ma20 is not None else None,
            "vol_ma20_start_date": str(df.iloc[-20]["date"]) if vol_ma20 is not None else None,
        }

    def execute(self, symbol: str, as_of_date: str | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> EvidenceBundle:
        target_date = self.normalize_as_of_date(as_of_date)

        try:
            raw_df = self._kline_fetcher(symbol, target_date, lookback_days)
        except Exception as exc:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="error",
                is_degraded=True,
                degrade_reason=f"行情接口调用异常: {exc}",
                error_message=str(exc),
            )

        if raw_df is None or raw_df.empty:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="error",
                is_degraded=True,
                degrade_reason="未获取到有效K线数据",
                error_message="empty kline dataframe",
            )

        # 严格时间锚点过滤：杜绝晚于 as_of_date 的未来数据
        df = raw_df[raw_df["date"] <= target_date].copy()
        df = df.sort_values("date").reset_index(drop=True)

        if df.empty:
            return self.build_bundle(
                symbol=symbol,
                as_of_date=target_date,
                evidences=[],
                status="error",
                is_degraded=True,
                degrade_reason=f"基准日 {target_date} 及之前无可用交易日数据",
                error_message="no data before as_of_date",
            )

        metrics = self._compute_metrics(df)
        if "volume_unit" in raw_df.attrs:
            metrics["volume_unit"] = raw_df.attrs["volume_unit"]
        is_short_history = metrics["bars_count"] < 20
        degrade_reason = (
            f"可用交易日仅 {metrics['bars_count']} 天（少于20日），部分中短期指标参考性受限"
            if is_short_history
            else None
        )

        ret_5d_str = f"{metrics['ret_5d_pct']}%" if metrics["ret_5d_pct"] is not None else "N/A"
        ret_20d_str = f"{metrics['ret_20d_pct']}%" if metrics["ret_20d_pct"] is not None else "N/A"
        ma_str = (
            f"MA5={metrics['ma5']}, MA20={metrics['ma20']}, MA60={metrics['ma60']}"
            if metrics["ma60"] is not None
            else f"MA5={metrics['ma5']}, MA20={metrics['ma20']}"
        )
        dd_str = f"{metrics['max_drawdown_60d_pct']}%" if metrics["max_drawdown_60d_pct"] is not None else "N/A"
        vol_ratio_str = f"{metrics['vol_ratio_20d']}x" if metrics["vol_ratio_20d"] is not None else "N/A"

        summary = (
            f"截至 {metrics['latest_trade_date']} 收盘价 {metrics['close']}，"
            f"近5日涨跌幅 {ret_5d_str}，近20日涨跌幅 {ret_20d_str}；"
            f"均线状态：{metrics['ma_alignment']}（{ma_str}）；"
            f"最近 {min(metrics['bars_count'], 60)} 个交易日最大回撤 {dd_str}"
            f"（区间 {metrics['low_60d']} ~ {metrics['high_60d']}），"
            f"当日量能为20日均量的 {vol_ratio_str}。"
        )

        ev_item = self.build_evidence(
            symbol=symbol,
            title=f"{symbol} 量价与均线结构（截至 {metrics['latest_trade_date']}）",
            summary=summary,
            source=raw_df.attrs.get("source", self._source_name),
            published_at=metrics["latest_trade_date"],
            payload=metrics,
            is_degraded=is_short_history,
            degrade_reason=degrade_reason,
        )

        return self.build_bundle(
            symbol=symbol,
            as_of_date=target_date,
            evidences=[ev_item],
            status="ok",
            is_degraded=is_short_history,
            degrade_reason=degrade_reason,
        )
