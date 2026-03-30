import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.messages import HumanMessage, SystemMessage
# 导入你的四个 MCP 工具
from tools_mcp import (
    get_fundamental_data, 
    get_technical_data, 
    get_valuation_data, 
    get_news_sentiment
)

# ==========================================
# 1. 初始化大脑 (DeepSeek) 与 工具箱
# ==========================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-bb49ff0b810a4465b6d274683d52e8a9") 

# llm = ChatOpenAI(
#     model="deepseek-chat", 
#     api_key=DEEPSEEK_API_KEY, 
#     base_url="https://api.deepseek.com",
#     max_tokens=2048,
#     temperature=0.2 
# )
llm = ChatOpenAI(
    model="./qwen_model",  # 这里必须与你启动 vLLM 时的 --model 参数完全一致
    api_key="EMPTY",       # 本地 vLLM 不需要真实的 API Key
    base_url="http://localhost:8000/v1", # 指向你本地一直在后台运行的 vLLM 服务
    max_tokens=2048,
    temperature=0.2 
)
# 告诉 Agent 它拥有哪些武器
tools = [get_fundamental_data, get_technical_data, get_valuation_data, get_news_sentiment]

# ==========================================
# 2. 设定 ReAct 系统提示词 (专治 7B 模型偷懒版)
# ==========================================
# ==========================================
# 2. 设定 ReAct 系统提示词 (防代码幻觉终极版)
# ==========================================
# ==========================================
# 2. 设定 ReAct 系统提示词 (强行注入底层调用协议)
# ==========================================
system_prompt = """
你是一个严谨的量化股票投资顾问 Agent。
你的首要任务是获取实时数据。为了触发底层的 API，你必须抛弃人类的自然语言，使用系统规定的 XML 协议与底层通信。

【🚨 核心指令：如何调用工具】
当你想调用工具时，严禁使用自然语言解释！你必须且只能输出如下的 XML 格式代码块：

<tool_call>
{"name": "工具名称", "arguments": {"stock_code": "股票代码"}}
</tool_call>

【你可用的工具列表】
1. get_fundamental_data (基本面)
2. get_technical_data (技术面)
3. get_valuation_data (估值面)
4. get_news_sentiment (消息面)

【执行工作流】
1. 第一步：收到用户给的股票代码后，直接输出 <tool_call> 调用基本面工具，不要说任何废话！
2. 第二步：系统会把真实数据返回给你，你再继续用 <tool_call> 调用下一个工具。
3. 第三步：只有在你收齐了所有 4 个维度的真实数据后，才能开始写最终的 Markdown 研报。研报中必须使用你刚刚获取到的真实数字！
"""

# ==========================================
# 3. 创建 ReAct Agent
# ==========================================
# create_react_agent 会自动为你处理 Thought -> Action -> Observation 循环
agent_executor = create_react_agent(llm, tools)

# ==========================================
# 4. 运行并保存 Markdown 到本地
# ==========================================
if __name__ == "__main__":
    stock_query = "600519"
    print(f"=== 开始唤醒 ReAct Agent 分析 {stock_query} ===\n")
    
    # 构造用户的输入消息
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请帮我全面分析A股股票 {stock_query}")
        ]
    }
    
    # 流式输出，观察 Agent 的“内心独白”和工具调用过程
    final_report = ""
    for chunk in agent_executor.stream(inputs, stream_mode="values"):
        message = chunk["messages"][-1]
        
        # 打印 Agent 调用的工具或思考过程
        if message.type == "ai" and message.tool_calls:
            for tool_call in message.tool_calls:
                print(f"🤔 [Agent 思考]: 决定调用工具 `{tool_call['name']}`，参数: {tool_call['args']}")
        elif message.type == "tool":
            print(f"✅ [系统反馈]: 工具 `{message.name}` 数据获取完成。")
        elif message.type == "ai" and not message.tool_calls:
            # 最终生成的文本
            final_report = message.content

    print("\n=== 最终研报生成完毕 ===\n")
    print(final_report)
    
    # --- 核心：保存到本地 Markdown 文件 ---
    output_filename = f"Report_{stock_query}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_report)
    
    print(f"\n📁 报告已成功保存至本地文件: {output_filename}")