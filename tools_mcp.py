import baostock as bs
import pandas as pd
import datetime
import requests
import json
import akshare as ak
import threading
import os
import time
import socket
from contextlib import contextmanager
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    import socks
except Exception:
    socks = None

# ==========================================
# 全局锁：保护 BaoStock 这种非线程安全的单例库
# ==========================================
bs_lock = threading.Lock()
BAOSTOCK_ENABLED = os.getenv("USE_BAOSTOCK", "0").lower() in ("1", "true", "yes")
BAOSTOCK_RETRIES = int(os.getenv("BAOSTOCK_RETRIES", "2"))
BAOSTOCK_RETRY_DELAY = float(os.getenv("BAOSTOCK_RETRY_DELAY", "1.5"))
BAOSTOCK_PROXY = os.getenv("BAOSTOCK_SOCKS5_PROXY", "").strip()
TOOL_CACHE_TTL_SECONDS = int(os.getenv("TOOL_CACHE_TTL_SECONDS", "30"))
tool_cache_lock = threading.Lock()
tool_cache: dict[str, tuple[float, dict]] = {}

# ==========================================
# 1. 严格的输入 Schema & 格式化工具
# ==========================================
class StockInputSchema(BaseModel):
    stock_code: str = Field(..., description="A股股票代码，例如 '600519'")

def format_bs_code(code: str) -> str:
    """将纯数字代码转换为 baostock 需要的格式 (如 sh.600519 或 sz.000001)"""
    code = ''.join(filter(str.isdigit, code))[:6]
    return f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"


def _safe_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().replace("%", "")
        if value in ("", "-", "--"):
            return None
        return float(value)
    except Exception:
        return None


def _extract_from_row(row, keys):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def _cache_key(tool_name: str, stock_code: str) -> str:
    pure_code = "".join(filter(str.isdigit, stock_code))[:6]
    return f"{tool_name}:{pure_code}"


def _cache_get(tool_name: str, stock_code: str):
    key = _cache_key(tool_name, stock_code)
    with tool_cache_lock:
        item = tool_cache.get(key)
        if not item:
            return None
        ts, value = item
        if (time.time() - ts) > TOOL_CACHE_TTL_SECONDS:
            tool_cache.pop(key, None)
            return None
        return dict(value)


def _cache_set(tool_name: str, stock_code: str, value: dict):
    key = _cache_key(tool_name, stock_code)
    with tool_cache_lock:
        tool_cache[key] = (time.time(), dict(value))


def _parse_socks_proxy(proxy_url: str):
    parsed = urlparse(proxy_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("socks5", "socks5h"):
        raise ValueError("BAOSTOCK_SOCKS5_PROXY 仅支持 socks5:// 或 socks5h://")
    if not parsed.hostname or not parsed.port:
        raise ValueError("BAOSTOCK_SOCKS5_PROXY 缺少 host 或 port")
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password,
        "rdns": scheme == "socks5h",
    }


@contextmanager
def _baostock_proxy_context():
    if not BAOSTOCK_PROXY:
        yield
        return
    if socks is None:
        raise RuntimeError("未安装PySocks，无法启用BAOSTOCK_SOCKS5_PROXY")
    cfg = _parse_socks_proxy(BAOSTOCK_PROXY)
    original_socket = socket.socket
    socks.set_default_proxy(
        proxy_type=socks.SOCKS5,
        addr=cfg["host"],
        port=cfg["port"],
        username=cfg["username"],
        password=cfg["password"],
        rdns=cfg["rdns"],
    )
    socket.socket = socks.socksocket
    try:
        yield
    finally:
        socket.socket = original_socket
        socks.set_default_proxy()


def _query_baostock(bs_code: str, fields: str, start_date: str, end_date: str):
    if not BAOSTOCK_ENABLED:
        raise RuntimeError("USE_BAOSTOCK未启用")
    last_error = "unknown"
    for _ in range(max(1, BAOSTOCK_RETRIES + 1)):
        try:
            with bs_lock:
                with _baostock_proxy_context():
                    lg = bs.login()
                    if getattr(lg, "error_code", "0") != "0":
                        raise RuntimeError(getattr(lg, "error_msg", "baostock login failed"))
                    rs = bs.query_history_k_data_plus(
                        bs_code,
                        fields,
                        start_date=start_date,
                        end_date=end_date,
                        frequency="d",
                        adjustflag="3",
                    )
                    if getattr(rs, "error_code", "0") != "0":
                        raise RuntimeError(getattr(rs, "error_msg", "baostock query failed"))
                    data_list = []
                    while (rs.error_code == "0") and rs.next():
                        data_list.append(rs.get_row_data())
                    return data_list, rs.fields
        except Exception as exc:
            last_error = str(exc)
            time.sleep(BAOSTOCK_RETRY_DELAY)
        finally:
            try:
                bs.logout()
            except Exception:
                pass
    raise RuntimeError(last_error)


def _fallback_fundamental_by_ak(stock_code: str):
    pure_code = ''.join(filter(str.isdigit, stock_code))[:6]
    spot_df = ak.stock_zh_a_spot_em()
    row_df = spot_df[spot_df["代码"] == pure_code]
    if row_df.empty:
        return {"status": "error", "message": f"AkShare现货中找不到代码 {stock_code}"}
    row = row_df.iloc[0]
    current_price = _safe_float(_extract_from_row(row, ["最新价", "最新"]))
    pe_ratio = _safe_float(_extract_from_row(row, ["市盈率-动态", "市盈率", "PE"]))
    pb_ratio = _safe_float(_extract_from_row(row, ["市净率", "PB"]))
    turnover_rate = _safe_float(_extract_from_row(row, ["换手率", "换手率(%)"]))
    if current_price is None:
        return {"status": "error", "message": f"AkShare现货数据不完整: {stock_code}"}
    return {
        "status": "success",
        "current_price": current_price,
        "pe_ratio_ttm": pe_ratio,
        "pb_ratio": pb_ratio,
        "turnover_rate": turnover_rate,
        "data_source": "akshare_spot",
    }


def _fallback_technical_by_ak(stock_code: str):
    pure_code = ''.join(filter(str.isdigit, stock_code))[:6]
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=180)).strftime("%Y%m%d")
    hist = ak.stock_zh_a_hist(symbol=pure_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    if hist is None or hist.empty:
        return {"status": "error", "message": f"AkShare历史行情为空: {stock_code}"}
    close_col = "收盘" if "收盘" in hist.columns else ("close" if "close" in hist.columns else None)
    if close_col is None:
        return {"status": "error", "message": "AkShare历史行情缺少收盘价列"}
    close_series = pd.to_numeric(hist[close_col], errors="coerce").dropna()
    if len(close_series) < 50:
        return {"status": "error", "message": "AkShare历史交易日不足50天"}
    current_price = float(close_series.iloc[-1])
    ma50 = float(close_series.rolling(window=50).mean().iloc[-1])
    return {
        "status": "success",
        "current_price": current_price,
        "fifty_day_average": round(ma50, 2),
        "data_source": "akshare_hist",
    }


def _fallback_valuation_by_ak(stock_code: str):
    pure_code = ''.join(filter(str.isdigit, stock_code))[:6]
    symbol_candidates = [pure_code, format_bs_code(pure_code).replace(".", "")]
    indicator_df = None
    for symbol in symbol_candidates:
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is not None and not df.empty:
                indicator_df = df
                break
        except Exception:
            continue
    if indicator_df is not None:
        pe_col = next((c for c in indicator_df.columns if ("pe" in c.lower() or "市盈率" in c)), None)
        date_col = next((c for c in indicator_df.columns if ("date" in c.lower() or "日期" in c)), None)
        if pe_col:
            df = indicator_df.copy()
            df[pe_col] = pd.to_numeric(df[pe_col], errors="coerce")
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                cutoff = datetime.datetime.now() - datetime.timedelta(days=1095)
                df = df[df[date_col] >= cutoff]
            pe_list = [x for x in df[pe_col].dropna().tolist() if x > 0]
            if pe_list:
                current_pe = float(pe_list[-1])
                sorted_pe = sorted(pe_list)
                position = sorted_pe.index(current_pe)
                percentile = (position / len(sorted_pe)) * 100
                if percentile < 30:
                    valuation_status = "低估区间，具备安全边际"
                elif percentile > 70:
                    valuation_status = "高估区间，存在杀估值风险"
                else:
                    valuation_status = "合理区间，估值中性"
                summary = f"当前动态市盈率(PE-TTM)为 {current_pe:.2f}，处于近三年 {percentile:.1f}% 的历史分位，整体估值处于{valuation_status}。"
                return {
                    "status": "success",
                    "current_pe": current_pe,
                    "pe_percentile_3y": round(percentile, 2),
                    "summary": summary,
                    "data_source": "akshare_indicator",
                }
    fundamental = _fallback_fundamental_by_ak(stock_code)
    pe_value = _safe_float(fundamental.get("pe_ratio_ttm")) if isinstance(fundamental, dict) else None
    if pe_value and pe_value > 0:
        return {
            "status": "success",
            "current_pe": pe_value,
            "pe_percentile_3y": 50.0,
            "summary": f"当前动态市盈率(PE-TTM)约为 {pe_value:.2f}，因历史估值不可用暂按中位分位(50%)处理。",
            "data_source": "akshare_spot_estimated",
        }
    return {"status": "error", "message": f"AkShare估值数据不可用: {stock_code}"}


def _extract_json_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

# ==========================================
# 2. 基本面 MCP 工具 (基于 BaoStock)
# ==========================================
@tool("get_fundamental_data", args_schema=StockInputSchema)
def get_fundamental_data(stock_code: str) -> dict:
    """获取指定A股股票的基本面核心数据（动态市盈率PE、市净率PB等）。"""
    cached = _cache_get("get_fundamental_data", stock_code)
    if cached:
        cached["from_cache"] = True
        return cached
    bs_code = format_bs_code(stock_code)
    
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    
    try:
        print("[基本面工具] 🔐 开始请求 BaoStock...")
        data_list, _ = _query_baostock(bs_code, "date,code,close,peTTM,pbMRQ,turn", start_date, end_date)
        print("[基本面工具] 🔓 BaoStock 请求完成。")
        if not data_list:
            raise RuntimeError(f"BaoStock返回空数据: {stock_code}")
        latest = data_list[-1]
        result = {
            "status": "success",
            "current_price": _safe_float(latest[2]),
            "pe_ratio_ttm": _safe_float(latest[3]),
            "pb_ratio": _safe_float(latest[4]),
            "turnover_rate": _safe_float(latest[5]),
            "data_source": "baostock",
        }
        _cache_set("get_fundamental_data", stock_code, result)
        return result
    except Exception as e:
        print(f"[基本面工具] BaoStock失败，尝试AkShare兜底: {e}")
        try:
            result = _fallback_fundamental_by_ak(stock_code)
            if result.get("status") == "success":
                _cache_set("get_fundamental_data", stock_code, result)
            return result
        except Exception as fallback_e:
            return {"status": "error", "message": f"基本面数据获取崩溃: {str(e)} | AkShare兜底失败: {str(fallback_e)}"}
# ==========================================
# 3. 技术面 MCP 工具 (基于 BaoStock)
# ==========================================
@tool("get_technical_data", args_schema=StockInputSchema)
def get_technical_data(stock_code: str) -> dict:
    """获取指定A股股票的技术面数据（均线系统）。"""
    cached = _cache_get("get_technical_data", stock_code)
    if cached:
        cached["from_cache"] = True
        return cached
    bs_code = format_bs_code(stock_code)
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y-%m-%d")
    
    try:
        print("[技术面工具] 🔐 开始请求 BaoStock...")
        data_list, fields = _query_baostock(bs_code, "date,close", start_date, end_date)
        print("[技术面工具] 🔓 BaoStock 请求完成。")
        if len(data_list) < 50:
            raise RuntimeError("BaoStock历史交易日不足50天")
        df = pd.DataFrame(data_list, columns=fields)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        close = df["close"].dropna()
        if len(close) < 50:
            raise RuntimeError("BaoStock close列有效数据不足50天")
        current_price = float(close.iloc[-1])
        ma50 = float(close.rolling(window=50).mean().iloc[-1])
        result = {
            "status": "success",
            "current_price": current_price,
            "fifty_day_average": round(ma50, 2),
            "data_source": "baostock",
        }
        _cache_set("get_technical_data", stock_code, result)
        return result
    except Exception as e:
        print(f"[技术面工具] BaoStock失败，尝试AkShare兜底: {e}")
        try:
            result = _fallback_technical_by_ak(stock_code)
            if result.get("status") == "success":
                _cache_set("get_technical_data", stock_code, result)
            return result
        except Exception as fallback_e:
            return {"status": "error", "message": f"技术面数据获取崩溃: {str(e)} | AkShare兜底失败: {str(fallback_e)}"}

# ==========================================
# 4. 新闻面分析工具 (请求 AkShare 和本地 Qwen-LoRA)
# ==========================================
@tool("get_news_sentiment", args_schema=StockInputSchema)
def get_news_sentiment(stock_code: str) -> dict:
    """获取指定A股股票的最新真实新闻舆情与风险标签。"""
    cached = _cache_get("get_news_sentiment", stock_code)
    if cached:
        cached["from_cache"] = True
        return cached
    pure_code = ''.join(filter(str.isdigit, stock_code))[:6]
    print(f"-> [工具后台] 正在从全网抓取 {pure_code} 的最新真实新闻...")
    
    real_news_text = ""
    try:
        news_df = ak.stock_news_em(symbol=pure_code)
        if not news_df.empty:
            top_news = news_df.head(3)
            news_list = []
            for _, row in top_news.iterrows():
                title = row.get("新闻标题", "无标题")
                content = row.get("新闻内容", "无内容摘要")
                time_str = row.get("发布时间", "")
                news_list.append(f"【{time_str}】标题：{title}\n摘要：{content}")
            real_news_text = "\n\n".join(news_list)
            print(f"-> [工具后台] 成功抓取到 {len(top_news)} 条最新新闻，准备喂给 Qwen-LoRA。")
        else:
            real_news_text = f"近期暂无关于 {stock_code} 的重大新闻披露。"
            print("-> [工具后台] 未抓取到最新新闻。")
    except Exception as e:
        print(f"\n⚠️ [系统告警] 真实新闻抓取失败 ({str(e)})。降级为无新闻状态。\n")
        real_news_text = f"近期暂无关于 {stock_code} 的重大新闻披露。"

    print(f"-> [工具后台] 正在请求本地 Qwen2.5-7B(LoRA) 分析新闻情感...")
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "finance_news", 
                "messages": [
                    {
                        "role": "system", 
                        "content": "你是一个专业的金融风控分析师。请分析输入的新闻或资讯，严格以 JSON 格式输出情感极性(sentiment)、得分(score)以及风险/利好标签(risk_tags)。如果输入提示无重大新闻，请输出中性的评分和空标签。"
                    },
                    {
                        "role": "user", 
                        "content": f"分析以下金融资讯：\n{real_news_text}"
                    }
                ],
                "temperature": 0.1 
            },
            timeout=90,
        )
        response.raise_for_status()
        result_str = response.json()["choices"][0]["message"]["content"]
        result_json = json.loads(_extract_json_text(result_str))
        result_json["status"] = "success"
        result_json["original_news_summary"] = real_news_text[:500] + "..." if len(real_news_text) > 500 else real_news_text
        print(f"-> [工具后台] 模型抽取结果: {result_json}")
        _cache_set("get_news_sentiment", stock_code, result_json)
        return result_json
    except Exception as e:
        result = {
            "status": "success",
            "sentiment": "neutral",
            "score": 0.0,
            "risk_tags": [],
            "original_news_summary": real_news_text[:500] + "..." if len(real_news_text) > 500 else real_news_text,
            "degraded": True,
            "message": f"新闻情绪模型解析失败，已降级为中性结果: {str(e)}",
        }
        _cache_set("get_news_sentiment", stock_code, result)
        return result

# ==========================================
# 5. 估值面 MCP 工具 (基于 BaoStock)
# ==========================================
@tool("get_valuation_data", args_schema=StockInputSchema)
def get_valuation_data(stock_code: str) -> dict:
    """获取指定A股股票的真实估值面数据（计算近三年市盈率PE的历史分位）。"""
    cached = _cache_get("get_valuation_data", stock_code)
    if cached:
        cached["from_cache"] = True
        return cached
    bs_code = format_bs_code(stock_code)
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=1095)).strftime("%Y-%m-%d")
    
    try:
        print("[估值面工具] 🔐 开始请求 BaoStock...")
        data_list, _ = _query_baostock(bs_code, "date,peTTM", start_date, end_date)
        print("[估值面工具] 🔓 BaoStock 请求完成。")
        pe_list = []
        for row in data_list:
            pe = _safe_float(row[1] if len(row) > 1 else None)
            if pe and pe > 0:
                pe_list.append(pe)
        if not pe_list:
            raise RuntimeError("BaoStock没有可用历史PE")
        current_pe = pe_list[-1]
        sorted_pe_list = sorted(pe_list)
        position = sorted_pe_list.index(current_pe)
        percentile = (position / len(sorted_pe_list)) * 100
        if percentile < 30:
            valuation_status = "低估区间，具备安全边际"
        elif percentile > 70:
            valuation_status = "高估区间，存在杀估值风险"
        else:
            valuation_status = "合理区间，估值中性"
        summary = f"当前动态市盈率(PE-TTM)为 {current_pe:.2f}，处于近三年 {percentile:.1f}% 的历史分位，整体估值处于{valuation_status}。"
        result = {
            "status": "success",
            "current_pe": current_pe,
            "pe_percentile_3y": round(percentile, 2),
            "summary": summary,
            "data_source": "baostock",
        }
        _cache_set("get_valuation_data", stock_code, result)
        return result
    except Exception as e:
        print(f"[估值面工具] BaoStock失败，尝试AkShare兜底: {e}")
        try:
            result = _fallback_valuation_by_ak(stock_code)
            if result.get("status") == "success":
                _cache_set("get_valuation_data", stock_code, result)
            return result
        except Exception as fallback_e:
            return {"status": "error", "message": f"估值数据获取崩溃: {str(e)} | AkShare兜底失败: {str(fallback_e)}"}
