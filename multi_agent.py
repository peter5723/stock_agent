import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
import json
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# 导入四个工具（保持不变）
from tools_mcp import (
    get_fundamental_data,
    get_technical_data,
    get_valuation_data,
    get_news_sentiment,
)

# =========================================================
# 1) 初始化 LLM（保持与现有 ChatOpenAI(vLLM) 初始化一致）
# =========================================================
llm = ChatOpenAI(
    model="./qwen_model",               # 必须与 vLLM --model 一致
    api_key="EMPTY",                    # 本地 vLLM 无需真实 Key
    base_url="http://localhost:8000/v1",  # 指向本地 vLLM
    max_tokens=2048,
    temperature=0.2,
)

# =========================================================
# 2) 定义 State（图的共享状态）
# =========================================================
class StockState(TypedDict):
    # 输入
    stock_code: str
    # 4 个专家节点的产出
    fundamental_data: str
    technical_data: str
    news_data: str
    valuation_data: str
    # 主编节点产出
    final_report: str


# =========================================================
# 3) 专家节点：直接调用工具（并行执行）
#    注意：工具返回为 dict，这里统一转为 JSON 字符串存入 state
# =========================================================
def fundamental_agent(state: StockState) -> StockState:
    code = state["stock_code"]
    res = get_fundamental_data.invoke({"stock_code": code})
    return {"fundamental_data": json.dumps(res, ensure_ascii=False)}


def technical_agent(state: StockState) -> StockState:
    code = state["stock_code"]
    res = get_technical_data.invoke({"stock_code": code})
    return {"technical_data": json.dumps(res, ensure_ascii=False)}


def news_agent(state: StockState) -> StockState:
    code = state["stock_code"]
    res = get_news_sentiment.invoke({"stock_code": code})
    return {"news_data": json.dumps(res, ensure_ascii=False)}


def valuation_agent(state: StockState) -> StockState:
    code = state["stock_code"]
    res = get_valuation_data.invoke({"stock_code": code})
    return {"valuation_data": json.dumps(res, ensure_ascii=False)}


# =========================================================
# 4) 主编节点：汇总 4 维数据，调用 LLM 生成最终 Markdown 研报
# =========================================================
CHIEF_SYSTEM_PROMPT = (
    "你是一个客观冷酷的量化基金首席风控官和投研主编。"
    "请综合基本面、技术面、消息面、估值面数据，严格以 Markdown 格式撰写股票研报。"
    "必须包含：核心投资结论、基本面与估值、技术面趋势、舆情与风控、风险提示。"
    "严禁捏造未提供的数据。"
)


def chief_editor_agent(state: StockState) -> StockState:
    code = state["stock_code"]

    # 将四个维度的字符串转回可读 JSON 片段，便于 LLM 参考
    def _pretty(text: str) -> str:
        try:
            obj = json.loads(text) if text else {}
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return text or ""

    fundamental = _pretty(state.get("fundamental_data", ""))
    technical = _pretty(state.get("technical_data", ""))
    news = _pretty(state.get("news_data", ""))
    valuation = _pretty(state.get("valuation_data", ""))

    user_prompt = (
        f"标的股票代码：{code}\n\n"
        f"【基本面数据】\n{fundamental}\n\n"
        f"【技术面数据】\n{technical}\n\n"
        f"【消息面数据】\n{news}\n\n"
        f"【估值面数据】\n{valuation}\n\n"
        "请基于已给出的数据撰写 Markdown 研报，严格遵守系统提示中的结构与限制。"
    )

    messages = [
        SystemMessage(content=CHIEF_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    resp = llm.invoke(messages)
    content = getattr(resp, "content", "") or ""
    return {"final_report": content}


# =========================================================
# 5) 构建 LangGraph：4 并行专家节点 + 1 汇聚主编节点
# =========================================================
def build_app():
    graph = StateGraph(StockState)
    # 注册节点
    graph.add_node("fundamental_agent", fundamental_agent)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("news_agent", news_agent)
    graph.add_node("valuation_agent", valuation_agent)
    graph.add_node("chief_editor_agent", chief_editor_agent)

    # Fan-out：从 START 并行触发 4 个专家节点
    graph.add_edge(START, "fundamental_agent")
    graph.add_edge(START, "technical_agent")
    graph.add_edge(START, "news_agent")
    graph.add_edge(START, "valuation_agent")

    # Fan-in：4 个专家节点全部汇聚到主编节点
    graph.add_edge("fundamental_agent", "chief_editor_agent")
    graph.add_edge("technical_agent", "chief_editor_agent")
    graph.add_edge("news_agent", "chief_editor_agent")
    graph.add_edge("valuation_agent", "chief_editor_agent")

    # 结束
    graph.add_edge("chief_editor_agent", END)
    return graph.compile()


# =========================================================
# 6) 运行并保存 Markdown 到本地（查询 600519）
# =========================================================
if __name__ == "__main__":
    stock_query = "600519"
    print(f"=== 并行多智能体（Multi-Agent）启动：{stock_query} ===")

    app = build_app()
    # 运行图：输入仅需 stock_code，其余字段由节点写入
    result = app.invoke({"stock_code": stock_query})

    final_report = result.get("final_report", "")
    print("\n=== 最终研报（摘要） ===\n")
    print(final_report)

    output_filename = f"Report_{stock_query}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n📁 报告已成功保存至本地文件: {output_filename}")
