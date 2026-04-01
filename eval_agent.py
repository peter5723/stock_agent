import argparse
import concurrent.futures
import ast
import json
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

try:
    from multi_agent import build_app

    multi_agent_app = build_app()
except Exception:
    multi_agent_app = None

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
    parser.add_argument("--output-prefix", default=os.getenv("EVAL_OUTPUT_PREFIX", "agent_eval"))
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--suite", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--stocks", default="")
    parser.add_argument("--case-workers", type=int, default=4, help="并发评测多少个股票 Case")
    return parser.parse_args()


def resolve_stocks(args: argparse.Namespace) -> list[str]:
    if args.stocks:
        raw = [x.strip() for x in args.stocks.split(",") if x.strip()]
        return list(dict.fromkeys(raw))
    if args.suite == "full":
        return DEFAULT_STOCKS_FULL
    return DEFAULT_STOCKS_SMOKE


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
        ok = sentiment in {"positive", "negative", "neutral", "中性", "利好", "利空","积极", "消极"} and isinstance(score, (int, float))
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


def parse_state_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        try:
            obj = ast.literal_eval(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def build_tool_metrics_from_state(state: dict[str, Any], case: EvalCase) -> list[dict[str, Any]]:
    mapping = [
        ("get_fundamental_data", "fundamental_data"),
        ("get_technical_data", "technical_data"),
        ("get_news_sentiment", "news_data"),
        ("get_valuation_data", "valuation_data"),
    ]
    metrics: list[dict[str, Any]] = []
    for idx, (tool_name, state_key) in enumerate(mapping):
        tool_id = f"state_{idx}"
        tool_output = parse_state_dict(state.get(state_key))
        success = tool_output.get("status") == "success"
        consistency_ok, consistency_reason = validate_consistency(tool_name, tool_output) if tool_output else (False, "state缺失或解析失败")
        metrics.append(
            {
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "tool_latency_s": 0.0,
                "tool_success": success,
                "consistency_ok": consistency_ok,
                "consistency_reason": consistency_reason,
                "tool_output": tool_output,
            }
        )
    return metrics


def run_case(case: EvalCase, args: argparse.Namespace) -> dict[str, Any]:
    # 端到端评测：直接调用 multi_agent 的 LangGraph 图（不再手动构造 chat/completions / tool_calls）
    start_e2e = time.perf_counter()
    tool_call_metrics: list[dict[str, Any]] = []
    final_report = ""
    error_message = ""
    status = "Success"
    degraded_e2e = False
    error_stage = ""
    try:
        if multi_agent_app is None:
            raise RuntimeError("multi_agent_app 初始化失败：请检查 multi_agent.py 及其依赖是否可导入")

        # 运行多智能体图，返回 State（包含4个专家结果 + final_report）
        state = multi_agent_app.invoke({"stock_code": case.stock_code})

        # 1) 从 State 提取最终研报
        final_report = state.get("final_report", "") or ""

        # 2) 从 State 提取四个专家产出，计算工具成功率与一致性
        tool_call_metrics = build_tool_metrics_from_state(state, case)
    except Exception as exc:
        status = "Failed"
        error_message = str(exc)
        error_stage = "graph_invoke"
    e2e_latency = time.perf_counter() - start_e2e
    tool_total = len(tool_call_metrics)
    tool_success = sum(1 for x in tool_call_metrics if x["tool_success"])
    tool_consistency = sum(1 for x in tool_call_metrics if x["consistency_ok"])
    called_tools = {x["tool_name"] for x in tool_call_metrics if x.get("tool_output")}
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
        "model_first_latency_s": 0.0,
        "model_second_latency_s": 0.0,
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
    all_results: list[dict[str, Any]] = []

    # Case 级并发：同时评测多支股票
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.case_workers)) as executor:
        future_map = {executor.submit(run_case, case, args): case for case in cases}
        for future in concurrent.futures.as_completed(future_map):
            case = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "case_id": case.case_id,
                    "stock_code": case.stock_code,
                    "status": "Failed",
                    "error": str(exc),
                    "error_stage": "case_future",
                    "degraded_e2e": False,
                    "tool_calls_total": 0,
                    "tool_calls_success": 0,
                    "tool_success_rate": 0.0,
                    "data_consistency_rate": 0.0,
                    "expected_tool_coverage": 0.0,
                    "model_first_latency_s": 0.0,
                    "model_second_latency_s": 0.0,
                    "e2e_latency_s": 0.0,
                    "objective_score_40": 0,
                    "llm_score_60": 0,
                    "total_quality_score_100": 0,
                    "quality_missing_elements": [],
                    "judge_reason": "",
                    "final_report_preview": "",
                    "tool_call_metrics": [],
                }
            all_results.append(result)
            print(
                f"完成评测: {result['case_id']} | status={result['status']} | "
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
