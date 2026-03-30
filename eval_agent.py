import argparse
import concurrent.futures
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

@dataclass
class EvalCase:
    case_id: str
    stock_code: str
    query: str
    expected_tools: list[str]


DEFAULT_STOCKS_SMOKE = ["600519", "000001", "300750", "601138"]
DEFAULT_STOCKS_FULL = [
    "600519",
    "000001",
    "300750",
    "601138",
    "601318",
    "600036",
    "600276",
    "002415",
    "000858",
    "600809",
    "002594",
    "601012",
    "688981",
    "000333",
    "601088",
    "601166",
    "600030",
    "300059",
    "601688",
    "600837",
    "601211",
    "600196",
    "300015",
    "300122",
    "000661",
    "600887",
    "002714",
    "600900",
    "600886",
    "601899",
    "601600",
    "600426",
    "603799",
    "002475",
    "002241",
    "600745",
    "688111",
    "688012",
    "688256",
    "600150",
    "600031",
    "000768",
    "601989",
    "601390",
    "600048",
    "001979",
    "600188",
    "601225",
    "600570",
    "300124",
    "300308",
    "002230",
    "601877",
    "300274",
    "300450",
    "601669",
    "600460",
    "002129",
    "603986",
    "000063",
    "600050",
]


def build_eval_cases(stocks: list[str]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for code in stocks:
        cases.append(
            EvalCase(
                case_id=f"stock_{code}",
                stock_code=code,
                query=f"请对A股{code}给出完整投资研报，必须调用基本面、技术面、新闻面、估值面四类工具，并在结尾给出风险提示与结论。",
                expected_tools=[
                    "get_fundamental_data",
                    "get_technical_data",
                    "get_news_sentiment",
                    "get_valuation_data",
                ],
            )
        )
    return cases


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_fundamental_data",
            "description": "获取A股股票基本面核心数据",
            "parameters": {
                "type": "object",
                "properties": {"stock_code": {"type": "string", "description": "A股股票代码，例如600519"}},
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_data",
            "description": "获取A股股票技术面均线和价格数据",
            "parameters": {
                "type": "object",
                "properties": {"stock_code": {"type": "string", "description": "A股股票代码，例如600519"}},
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news_sentiment",
            "description": "获取A股股票新闻情绪与风险标签",
            "parameters": {
                "type": "object",
                "properties": {"stock_code": {"type": "string", "description": "A股股票代码，例如600519"}},
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_valuation_data",
            "description": "获取A股股票估值分位信息",
            "parameters": {
                "type": "object",
                "properties": {"stock_code": {"type": "string", "description": "A股股票代码，例如600519"}},
                "required": ["stock_code"],
            },
        },
    },
]

TOOL_MAP: dict[str, Any] = {}

JUDGE_PROMPT = """
你是一个客观冷酷的量化基金首席风控官。请对输入的 AI 股票研报进行质量打分（满分 60 分）。
仅评估以下两点：
1. 逻辑自洽性 (30分): 研报中的投资建议是否与给出的基本面、技术面、消息面数据存在严重矛盾？
2. 语言专业度 (30分): 金融术语使用是否精准？有无大模型常见的“抱歉”、“我无法获取”等废话或幻觉？
请严格以 JSON 格式输出，绝对不要包含任何其他文字或 Markdown 标记：
{"llm_score": 55, "reasoning": "逻辑严密，扣除5分因为..."}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent工具调用评测")
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "qwen2.5-7b-instruct"))
    parser.add_argument("--api-key", default=os.getenv("QWEN_API_KEY", "EMPTY"))
    parser.add_argument("--request-timeout", type=int, default=int(os.getenv("EVAL_TIMEOUT", "120")))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-prefix", default=os.getenv("EVAL_OUTPUT_PREFIX", "agent_eval"))
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--stocks", default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--disable-tool-fallback", action="store_true")
    parser.add_argument("--tool-workers", type=int, default=4)
    return parser.parse_args()


def resolve_stocks(args: argparse.Namespace) -> list[str]:
    if args.stocks:
        raw = [x.strip() for x in args.stocks.split(",") if x.strip()]
        return list(dict.fromkeys(raw))
    if args.suite == "full":
        return DEFAULT_STOCKS_FULL
    return DEFAULT_STOCKS_SMOKE


def load_tool_map() -> None:
    global TOOL_MAP
    from tools_mcp import get_fundamental_data, get_news_sentiment, get_technical_data, get_valuation_data

    TOOL_MAP = {
        "get_fundamental_data": get_fundamental_data,
        "get_technical_data": get_technical_data,
        "get_news_sentiment": get_news_sentiment,
        "get_valuation_data": get_valuation_data,
    }


def resolve_runtime_model(base_url: str, requested_model: str, api_key: str, timeout: int) -> str:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        ids = [x.get("id") for x in models if x.get("id")]
        if requested_model in ids:
            return requested_model
        return ids[0] if ids else requested_model
    except Exception:
        return requested_model


def normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    normalized = []
    for idx, item in enumerate(raw_tool_calls):
        if not isinstance(item, dict):
            continue
        function_obj = item.get("function") if isinstance(item.get("function"), dict) else {}
        tool_name = function_obj.get("name")
        if not tool_name:
            continue
        normalized.append(
            {
                "id": item.get("id") or f"tool_call_{idx}",
                "function": {
                    "name": tool_name,
                    "arguments": function_obj.get("arguments") or "{}",
                },
            }
        )
    return normalized


def objective_eval(report_text: str) -> dict[str, Any]:
    checks = {
        "has_fundamental": bool(re.search(r"市盈率|PE|市净率|PB|换手率", report_text, re.IGNORECASE)),
        "has_technical": bool(re.search(r"均线|价格|技术面", report_text)),
        "has_news": bool(re.search(r"新闻|资讯|消息|情绪", report_text)),
        "has_risk_warning": bool(re.search(r"风险", report_text)),
        "proper_markdown": bool(re.search(r"## 核心投资结论|核心投资结论", report_text)),
    }
    score = sum(8 for v in checks.values() if v)
    return {"objective_score": score, "details": checks}


def llm_subjective_eval(report_text: str) -> dict[str, Any]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not deepseek_key:
        return {"llm_score": 0, "reasoning": "未配置DEEPSEEK_API_KEY，跳过主观评测"}
    try:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"请对以下研报打分：\n\n{report_text}"},
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
        }
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {deepseek_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as exc:
        return {"llm_score": 0, "reasoning": f"评测解析失败: {exc}"}


def call_chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    timeout: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    max_retries: int = 2,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    start = time.perf_counter()
    last_exc: Exception | None = None
    for _ in range(max(1, max_retries + 1)):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 404:
                payload["model"] = resolve_runtime_model(base_url, payload["model"], api_key, timeout)
                response = requests.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=timeout,
                )
            response.raise_for_status()
            latency_s = time.perf_counter() - start
            return response.json(), latency_s
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"chat/completions请求失败: {last_exc}")


def invoke_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if not TOOL_MAP:
        load_tool_map()
    if name not in TOOL_MAP:
        return {"status": "error", "message": f"未注册工具: {name}"}
    result = TOOL_MAP[name].invoke(arguments)
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"status": "error", "message": f"工具返回了非JSON字符串: {result[:200]}"}
    return {"status": "error", "message": f"工具返回类型不支持: {type(result).__name__}"}


def validate_consistency(tool_name: str, tool_output: dict[str, Any]) -> tuple[bool, str]:
    if tool_output.get("status") != "success":
        return False, tool_output.get("message", "status非success")
    if tool_name == "get_fundamental_data":
        ok = (
            tool_output.get("current_price") is not None
            and (tool_output.get("current_price") or 0) > 0
            and tool_output.get("pe_ratio_ttm") is not None
            and tool_output.get("pb_ratio") is not None
        )
        return (True, "ok") if ok else (False, "基本面字段缺失或数值异常")
    if tool_name == "get_technical_data":
        ok = (tool_output.get("current_price") or 0) > 0 and (tool_output.get("fifty_day_average") or 0) > 0
        return (True, "ok") if ok else (False, "技术面字段缺失或数值异常")
    if tool_name == "get_news_sentiment":
        sentiment = str(tool_output.get("sentiment", "")).lower()
        score = tool_output.get("score")
        ok = sentiment in {"positive", "negative", "neutral", "中性", "利好", "利空"} and isinstance(score, (int, float))
        return (True, "ok") if ok else (False, "新闻情绪格式不符合预期")
    if tool_name == "get_valuation_data":
        percentile = tool_output.get("pe_percentile_3y")
        ok = isinstance(percentile, (int, float)) and 0 <= percentile <= 100 and (tool_output.get("current_pe") or 0) > 0
        return (True, "ok") if ok else (False, "估值字段缺失或分位越界")
    return True, "ok"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    rank = int(round((len(values) - 1) * p))
    return sorted(values)[rank]


def build_optimization_suggestions(summary: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    if summary["tool_success_rate"] < 0.9:
        suggestions.append("工具调用成功率偏低：建议为每个工具增加重试机制、超时降级和统一异常码。")
    if summary["data_consistency_rate"] < 0.9:
        suggestions.append("数据一致率偏低：建议在工具返回层做强校验与字段补全，并在模型提示词中固定字段约束。")
    if summary["e2e_latency_p90_s"] > 15:
        suggestions.append("端到端时延P90偏高：建议并行拉取可并行工具（新闻与估值）并缓存分钟级行情。")
    if not suggestions:
        suggestions.append("当前指标达标：建议继续做分场景压测并按股票池规模扩展测试样本。")
    return suggestions


def synthesize_report_from_tools(case: EvalCase, tool_call_metrics: list[dict[str, Any]], reason: str) -> str:
    tools = {x["tool_name"]: x["tool_output"] for x in tool_call_metrics}
    fundamental = tools.get("get_fundamental_data", {})
    technical = tools.get("get_technical_data", {})
    news = tools.get("get_news_sentiment", {})
    valuation = tools.get("get_valuation_data", {})
    lines = [
        "## 核心投资结论",
        f"- 标的：{case.stock_code}",
        f"- 当前价格：{fundamental.get('current_price', 'N/A')}",
        f"- PE(TTM)：{fundamental.get('pe_ratio_ttm', valuation.get('current_pe', 'N/A'))}",
        f"- PB：{fundamental.get('pb_ratio', 'N/A')}",
        f"- 50日均线：{technical.get('fifty_day_average', 'N/A')}",
        f"- 新闻情绪：{news.get('sentiment', 'N/A')}（score={news.get('score', 'N/A')}）",
        f"- 估值分位：{valuation.get('pe_percentile_3y', 'N/A')}%",
        "",
        "## 风险提示",
        f"- 本次为降级生成，原因：{reason}",
        "- 建议结合公告、财报和盘中量价变化做二次确认。",
    ]
    return "\n".join(lines)


def resolve_tool_workers(args: argparse.Namespace, tool_call_count: int) -> int:
    workers = max(1, min(args.tool_workers, tool_call_count))
    if os.getenv("BAOSTOCK_SOCKS5_PROXY", "").strip():
        return 1
    return workers


def run_case(case: EvalCase, args: argparse.Namespace) -> dict[str, Any]:
    start_e2e = time.perf_counter()
    system_prompt = "你是专业A股投研助手，必须基于工具返回数据生成结构化结论。"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.query},
    ]
    tool_call_metrics: list[dict[str, Any]] = []
    model_first_latency = 0.0
    model_second_latency = 0.0
    final_report = ""
    error_message = ""
    status = "Success"
    degraded_e2e = False
    error_stage = ""
    try:
        first_resp, model_first_latency = call_chat_completion(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            messages=messages,
            timeout=args.request_timeout,
            temperature=args.temperature,
            tools=TOOLS,
            max_retries=args.max_retries,
        )
        first_msg = first_resp["choices"][0]["message"]
        tool_calls = normalize_tool_calls(first_msg.get("tool_calls"))
        if not tool_calls and not args.disable_tool_fallback:
            tool_calls = [
                {
                    "id": f"fallback_{i}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"stock_code": case.stock_code}, ensure_ascii=False),
                    },
                }
                for i, name in enumerate(case.expected_tools)
            ]
        messages.append(
            {
                "role": "assistant",
                "content": first_msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )
        def execute_tool_call(tc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            tool_name = tc["function"]["name"]
            tool_id = tc["id"]
            try:
                tool_args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                tool_args = {"stock_code": case.stock_code}
            tool_start = time.perf_counter()
            output = invoke_tool(tool_name, tool_args)
            tool_latency = time.perf_counter() - tool_start
            consistency_ok, consistency_reason = validate_consistency(tool_name, output)
            success = output.get("status") == "success"
            metric = {
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "tool_latency_s": round(tool_latency, 4),
                "tool_success": success,
                "consistency_ok": consistency_ok,
                "consistency_reason": consistency_reason,
                "tool_output": output,
            }
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(output, ensure_ascii=False),
            }
            return metric, tool_msg

        if tool_calls:
            indexed_calls = list(enumerate(tool_calls))
            results_by_index: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
            max_workers = resolve_tool_workers(args, len(indexed_calls))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(execute_tool_call, tc): idx for idx, tc in indexed_calls}
                for future in concurrent.futures.as_completed(future_map):
                    idx = future_map[future]
                    results_by_index[idx] = future.result()
            for idx in range(len(indexed_calls)):
                metric, tool_msg = results_by_index[idx]
                tool_call_metrics.append(metric)
                messages.append(tool_msg)

        if tool_calls:
            try:
                second_resp, model_second_latency = call_chat_completion(
                    base_url=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    messages=messages,
                    timeout=args.request_timeout,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                final_report = second_resp["choices"][0]["message"].get("content") or ""
            except Exception as second_exc:
                degraded_e2e = True
                error_stage = "second_completion"
                error_message = str(second_exc)
                final_report = synthesize_report_from_tools(case, tool_call_metrics, error_message)
                status = "Success"
        else:
            final_report = first_msg.get("content") or ""
    except Exception as exc:
        status = "Failed"
        error_message = str(exc)
        error_stage = "first_completion_or_tools"
    e2e_latency = time.perf_counter() - start_e2e
    tool_total = len(tool_call_metrics)
    tool_success = sum(1 for x in tool_call_metrics if x["tool_success"])
    tool_consistency = sum(1 for x in tool_call_metrics if x["consistency_ok"])
    called_tools = {x["tool_name"] for x in tool_call_metrics}
    expected_tools = set(case.expected_tools)
    objective_result = objective_eval(final_report) if final_report else {"objective_score": 0, "details": {}}
    llm_result = llm_subjective_eval(final_report) if args.enable_llm_judge and final_report else {"llm_score": 0, "reasoning": ""}
    return {
        "case_id": case.case_id,
        "stock_code": case.stock_code,
        "status": status,
        "error": error_message,
        "error_stage": error_stage,
        "degraded_e2e": degraded_e2e,
        "tool_calls_total": tool_total,
        "tool_calls_success": tool_success,
        "tool_success_rate": (tool_success / tool_total) if tool_total else 0.0,
        "data_consistency_rate": (tool_consistency / tool_total) if tool_total else 0.0,
        "expected_tool_coverage": (len(called_tools & expected_tools) / len(expected_tools)) if expected_tools else 1.0,
        "model_first_latency_s": round(model_first_latency, 4),
        "model_second_latency_s": round(model_second_latency, 4),
        "e2e_latency_s": round(e2e_latency, 4),
        "objective_score_40": objective_result["objective_score"],
        "llm_score_60": llm_result.get("llm_score", 0),
        "total_quality_score_100": objective_result["objective_score"] + llm_result.get("llm_score", 0),
        "quality_missing_elements": [k for k, v in objective_result.get("details", {}).items() if not v],
        "judge_reason": llm_result.get("reasoning", ""),
        "final_report_preview": final_report[:500],
        "tool_call_metrics": tool_call_metrics,
    }


def run_evaluation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_stocks = resolve_stocks(args)
    cases = build_eval_cases(selected_stocks)
    print(f"评测股票数量: {len(selected_stocks)}")
    all_results = []
    for case in cases:
        print(f"开始评测: {case.case_id}")
        result = run_case(case, args)
        all_results.append(result)
        print(
            f"完成评测: {case.case_id} | status={result['status']} | "
            f"tool_success_rate={result['tool_success_rate']:.2%} | "
            f"consistency_rate={result['data_consistency_rate']:.2%} | "
            f"e2e={result['e2e_latency_s']:.2f}s"
        )
    success_cases = [x for x in all_results if x["status"] == "Success"]
    e2e_latencies = [x["e2e_latency_s"] for x in all_results]
    tool_success_all = [x["tool_success_rate"] for x in all_results]
    consistency_all = [x["data_consistency_rate"] for x in all_results]
    summary = {
        "total_cases": len(all_results),
        "success_cases": len(success_cases),
        "e2e_success_rate": (len(success_cases) / len(all_results)) if all_results else 0.0,
        "tool_success_rate": statistics.mean(tool_success_all) if tool_success_all else 0.0,
        "data_consistency_rate": statistics.mean(consistency_all) if consistency_all else 0.0,
        "e2e_latency_avg_s": statistics.mean(e2e_latencies) if e2e_latencies else 0.0,
        "e2e_latency_p50_s": percentile(e2e_latencies, 0.5),
        "e2e_latency_p90_s": percentile(e2e_latencies, 0.9),
        "e2e_latency_p99_s": percentile(e2e_latencies, 0.99),
    }
    summary["optimization_suggestions"] = build_optimization_suggestions(summary)
    return all_results, summary


def export_outputs(results: list[dict[str, Any]], summary: dict[str, Any], prefix: str) -> None:
    rows = []
    for r in results:
        row = dict(r)
        row["tool_call_metrics"] = json.dumps(r["tool_call_metrics"], ensure_ascii=False)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = f"{prefix}_results.csv"
    summary_path = f"{prefix}_summary.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print("\n评测完成")
    print(f"结果明细: {csv_path}")
    print(f"汇总报告: {summary_path}")
    print(
        f"端到端成功率: {summary['e2e_success_rate']:.2%} | "
        f"工具调用成功率: {summary['tool_success_rate']:.2%} | "
        f"数据一致率: {summary['data_consistency_rate']:.2%} | "
        f"端到端时延P90: {summary['e2e_latency_p90_s']:.2f}s"
    )


if __name__ == "__main__":
    cli_args = parse_args()
    eval_results, eval_summary = run_evaluation(cli_args)
    export_outputs(eval_results, eval_summary, cli_args.output_prefix)
