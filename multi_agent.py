import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,0.0.0.0"
import ast
from datetime import datetime
import json
import re
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

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

ALLOWED_AGENTS = ["fundamental_agent", "technical_agent", "news_agent", "valuation_agent"]


def _extract_tool_args(ai_msg: Any, stock_code: str) -> dict:
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if tool_calls and isinstance(tool_calls, list) and isinstance(tool_calls[0], dict):
        args = tool_calls[0].get("args")
        if isinstance(args, dict):
            return args
    return {"stock_code": stock_code}


# =========================================================
# 2) 定义 State（图的共享状态）
# =========================================================
class StockState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    stock_code: str
    active_agents: list[str]
    fundamental_data: str
    technical_data: str
    news_data: str
    valuation_data: str
    final_answer: str
    final_report: str


class SupervisorDecision(BaseModel):
    stock_code: str = Field(default="", description="A股股票代码，6位数字，例如300750；提取不到则返回空字符串")
    active_agents: list[str] = Field(
        default_factory=list,
        description='要唤醒的专家节点列表，只能从 ["fundamental_agent","technical_agent","news_agent","valuation_agent"] 中选择若干个',
    )


SUPERVISOR_SYSTEM_PROMPT = (
    "你是股票咨询客服系统的主管（Supervisor）。"
    "你需要从用户问题中提取A股股票代码（6位数字），并判断需要唤醒哪些专家来获取数据。"
    "可选专家只有：fundamental_agent, technical_agent, news_agent, valuation_agent。"
    "如果用户只问新闻，就只选 news_agent；如果问市盈率/估值，就选 fundamental_agent 和 valuation_agent；"
    "如果问趋势/均线/形态，就选 technical_agent；如果问综合分析，就选四个都选。"
    "输出必须严格为 JSON，且只包含字段 stock_code 和 active_agents，禁止输出其他文字。"
)


def _extract_stock_code_from_text(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", text or "")
    return m.group(1) if m else ""


def _heuristic_active_agents(user_query: str) -> list[str]:
    q = (user_query or "").lower()
    agents: list[str] = []
    if any(k in q for k in ["新闻", "舆情", "情绪", "利好", "利空", "公告", "爆雷", "负面", "正面"]):
        agents.append("news_agent")
    if any(k in q for k in ["技术", "走势", "均线", "macd", "k线", "支撑", "压力", "形态", "量能", "成交量"]):
        agents.append("technical_agent")
    if any(k in q for k in ["基本面", "营收", "利润", "roe", "现金流", "分红", "财务", "业绩"]):
        agents.append("fundamental_agent")
    if any(k in q for k in ["估值", "分位", "pe", "pb", "市盈率", "市净率"]):
        if "fundamental_agent" not in agents:
            agents.append("fundamental_agent")
        agents.append("valuation_agent")
    if not agents and any(k in q for k in ["全面", "综合", "研报", "分析", "怎么看"]):
        agents = ALLOWED_AGENTS[:]
    return list(dict.fromkeys([a for a in agents if a in ALLOWED_AGENTS]))


def supervisor_agent(state: StockState) -> StockState:
    preset_stock_code = state.get("stock_code", "") or ""

    messages = state.get("messages", []) or []
    if preset_stock_code and not messages:
        return {"stock_code": preset_stock_code, "active_agents": ALLOWED_AGENTS[:]}

    transcript_lines: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if isinstance(msg, HumanMessage):
            transcript_lines.append(f"用户: {content}")
        elif isinstance(msg, AIMessage):
            transcript_lines.append(f"助手: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else ""

    stock_code = ""
    active_agents: list[str] = []
    try:
        if hasattr(llm, "with_structured_output"):
            llm_router = llm.with_structured_output(SupervisorDecision)
            decision = llm_router.invoke(
                [
                    SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                    HumanMessage(content=transcript),
                ]
            )
            if isinstance(decision, dict):
                parsed = SupervisorDecision(**decision)
            else:
                parsed = decision
            stock_code = (parsed.stock_code or "").strip()
            active_agents = [x for x in (parsed.active_agents or []) if x in ALLOWED_AGENTS]
        else:
            resp = llm.invoke(
                [
                    SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                    HumanMessage(content=transcript),
                ]
            )
            text = getattr(resp, "content", "") or ""
            try:
                obj = json.loads(text)
            except Exception:
                obj = ast.literal_eval(text)
            parsed = SupervisorDecision(**obj) if isinstance(obj, dict) else SupervisorDecision()
            stock_code = (parsed.stock_code or "").strip()
            active_agents = [x for x in (parsed.active_agents or []) if x in ALLOWED_AGENTS]
    except Exception:
        stock_code = _extract_stock_code_from_text(transcript)
        active_agents = _heuristic_active_agents(transcript)

    if not stock_code:
        stock_code = _extract_stock_code_from_text(transcript)
    active_agents = list(dict.fromkeys([a for a in active_agents if a in ALLOWED_AGENTS]))
    if not active_agents:
        active_agents = _heuristic_active_agents(transcript)
    if not stock_code:
        active_agents = []

    return {"stock_code": stock_code, "active_agents": active_agents}


# =========================================================
# 3) 专家节点：大模型驱动的工具调用（并行执行）
# =========================================================
def fundamental_agent(state: StockState) -> StockState:
    llm_with_tools = llm.bind_tools([get_fundamental_data], tool_choice="get_fundamental_data")
    prompt = f"请获取股票 {state['stock_code']} 的基本面数据。"
    ai_msg = llm_with_tools.invoke([HumanMessage(content=prompt)])
    tool_args = _extract_tool_args(ai_msg, state["stock_code"])
    res = get_fundamental_data.invoke(tool_args)
    return {"fundamental_data": str(res)}


def technical_agent(state: StockState) -> StockState:
    llm_with_tools = llm.bind_tools([get_technical_data], tool_choice="get_technical_data")
    prompt = f"请获取股票 {state['stock_code']} 的技术面数据。"
    ai_msg = llm_with_tools.invoke([HumanMessage(content=prompt)])
    tool_args = _extract_tool_args(ai_msg, state["stock_code"])
    res = get_technical_data.invoke(tool_args)
    return {"technical_data": str(res)}


def news_agent(state: StockState) -> StockState:
    llm_with_tools = llm.bind_tools([get_news_sentiment], tool_choice="get_news_sentiment")
    prompt = f"请获取股票 {state['stock_code']} 的消息面（新闻情绪）数据。"
    ai_msg = llm_with_tools.invoke([HumanMessage(content=prompt)])
    tool_args = _extract_tool_args(ai_msg, state["stock_code"])
    res = get_news_sentiment.invoke(tool_args)
    return {"news_data": str(res)}


def valuation_agent(state: StockState) -> StockState:
    llm_with_tools = llm.bind_tools([get_valuation_data], tool_choice="get_valuation_data")
    prompt = f"请获取股票 {state['stock_code']} 的估值面数据。"
    ai_msg = llm_with_tools.invoke([HumanMessage(content=prompt)])
    tool_args = _extract_tool_args(ai_msg, state["stock_code"])
    res = get_valuation_data.invoke(tool_args)
    return {"valuation_data": str(res)}


# =========================================================
# 4) 客服节点：汇总被唤醒专家的结果，按用户问题生成回答
# =========================================================
SYNTHESIZER_SYSTEM_PROMPT = (
    "你是股票咨询客服。"
    "你需要结合已提供的数据回答用户问题，若某类数据为空则忽略。"
    "你必须把【工具数据】视为唯一事实来源：价格/市盈率/日期/新闻等只能来自工具数据或对话历史中明确给出的工具数据。"
    "严禁使用模型自带知识库补全数据，严禁编造或猜测。"
    "如果工具数据未提供某项信息，请直接说明“工具数据未提供”，不要给出具体数值或年份。"
    "如果用户要求“最近/最新”，以今天日期为准，仅基于工具数据给出结论。"
    "输出使用 Markdown，结构清晰，尽量简洁。"
)


def _sanitize_answer(answer: str, tool_context: str, today: str) -> str:
    if not isinstance(answer, str) or not answer.strip():
        return ""
    tool_text = tool_context or ""
    out = answer
    current_year = int(today[:4])

    def _mask_year(m: re.Match) -> str:
        y = int(m.group(0))
        if y < current_year and str(y) not in tool_text:
            return "（工具数据未提供对应年份信息）"
        return m.group(0)

    out = re.sub(r"\b(20\d{2})\b", _mask_year, out)

    def _mask_price(m: re.Match) -> str:
        s = m.group(0)
        if "current_price" not in tool_text and "当前价格" not in tool_text and "现价" not in tool_text:
            return re.sub(r"\d+(\.\d+)?", "（工具数据未提供价格）", s, count=1)
        return s

    out = re.sub(r"(当前价格|现价)[^。\n]*?\d+(\.\d+)?\s*元", _mask_price, out)
    return out

def synthesizer_agent(state: StockState) -> StockState:
    stock_code = state.get("stock_code", "") or ""
    active_agents = state.get("active_agents", []) or []
    history = state.get("messages", []) or []

    def _pretty(text: str) -> str:
        try:
            obj = json.loads(text) if text else {}
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            try:
                obj = ast.literal_eval(text) if text else {}
                return json.dumps(obj, ensure_ascii=False, indent=2) if isinstance(obj, dict) else (text or "")
            except Exception:
                return text or ""

    blocks: list[str] = []
    if stock_code:
        blocks.append(f"股票代码：{stock_code}")
    if active_agents:
        blocks.append(f"已唤醒专家：{', '.join(active_agents)}")
    else:
        blocks.append("未唤醒任何专家。")

    fundamental = state.get("fundamental_data", "") or ""
    technical = state.get("technical_data", "") or ""
    news = state.get("news_data", "") or ""
    valuation = state.get("valuation_data", "") or ""

    if fundamental:
        blocks.append(f"\n【基本面数据】\n{_pretty(fundamental)}")
    if valuation:
        blocks.append(f"\n【估值面数据】\n{_pretty(valuation)}")
    if technical:
        blocks.append(f"\n【技术面数据】\n{_pretty(technical)}")
    if news:
        blocks.append(f"\n【消息面数据】\n{_pretty(news)}")

    context = "\n".join(blocks)
    today = datetime.now().strftime("%Y-%m-%d")
    time_message = SystemMessage(content=f"今天日期：{today}")
    data_message = SystemMessage(content=f"【工具数据】（可能不完整，缺失则表示未获取）：\n{context}")
    resp = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            time_message,
            data_message,
            *history,
        ]
    )
    answer = getattr(resp, "content", "") or ""
    answer = _sanitize_answer(answer, context, today)
    return {"final_answer": answer, "final_report": answer, "messages": [AIMessage(content=answer)]}


# =========================================================
# 5) 构建 LangGraph：Supervisor Pattern（动态路由 + 并行专家 + 汇总客服）
# =========================================================
def build_app():
    workflow = StateGraph(StockState)
    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("fundamental_agent", fundamental_agent)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("news_agent", news_agent)
    workflow.add_node("valuation_agent", valuation_agent)
    workflow.add_node("synthesizer_agent", synthesizer_agent)

    workflow.add_edge(START, "supervisor_agent")

    def _route(state: StockState):
        agents = state.get("active_agents", []) or []
        agents = [a for a in agents if a in ALLOWED_AGENTS]
        return agents if agents else "synthesizer_agent"

    workflow.add_conditional_edges("supervisor_agent", _route)

    workflow.add_edge("fundamental_agent", "synthesizer_agent")
    workflow.add_edge("technical_agent", "synthesizer_agent")
    workflow.add_edge("news_agent", "synthesizer_agent")
    workflow.add_edge("valuation_agent", "synthesizer_agent")

    # 结束
    workflow.add_edge("synthesizer_agent", END)
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)
    return compiled


class _MultiAgentAppWrapper:
    def __init__(self, app):
        self._app = app

    def invoke(self, input: dict, config: dict | None = None, **kwargs):
        if config is None:
            config = {"configurable": {"thread_id": "default_session"}}
        else:
            configurable = config.get("configurable") if isinstance(config, dict) else None
            if not isinstance(configurable, dict):
                config = {"configurable": {"thread_id": "default_session"}}
            elif "thread_id" not in configurable:
                configurable["thread_id"] = "default_session"
        return self._app.invoke(input, config=config, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._app, name)


multi_agent_app = _MultiAgentAppWrapper(build_app())


# =========================================================
# 6) 运行测试（问答式）
# =========================================================
if __name__ == "__main__":
    print("=== 股票咨询多智能体助手已启动 (输入 'quit' 或 'exit' 退出) ===")
    config = {"configurable": {"thread_id": "session_001"}}
    while True:
        user_input = input("\n[你]: ")
        if user_input.lower() in ["quit", "exit"]:
            print("再见！")
            break
        if not user_input.strip():
            continue
        result = multi_agent_app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        print(f"\n[AI助手]:\n{result.get('final_answer', '')}")
