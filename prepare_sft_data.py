import json
from datasets import load_dataset

# 1. 设定 System Prompt (这与你在 LangGraph 中设定的一致，确保训练和推理同源)
SYSTEM_PROMPT = """你是一个专业的金融风控分析师。请分析输入的新闻或资讯，严格以 JSON 格式输出情感极性(sentiment)、得分(score)以及风险/利好标签(risk_tags)。格式示例：{"sentiment": "positive", "score": 0.8, "risk_tags": ["利好信号"]}"""

def convert_to_chatml(example):
    """
    将 FinGPT 的原始数据映射为 Qwen 需要的 ChatML 结构化 JSON 格式。
    原始 label 通常为：positive, negative, neutral
    """
    news_text = example.get('input', '') or example.get('text', '')
    original_label = str(example.get('output', '') or example.get('label', '')).lower()
    
    # 根据开源数据集的标签，人为构造我们需要的结构化输出
    if "positive" in original_label:
        sentiment = "positive"
        score = 0.8
        risk_tags = ["利好信号"]
    elif "negative" in original_label:
        sentiment = "negative"
        score = 0.2
        risk_tags = ["负面预警"]
    else:
        sentiment = "neutral"
        score = 0.5
        risk_tags = ["中性事件"]
        
    # 构造 Assistant 应该输出的 JSON 字符串
    assistant_json_str = json.dumps({
        "sentiment": sentiment,
        "score": score,
        "risk_tags": risk_tags
    }, ensure_ascii=False)

    # 拼装标准的 ChatML messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"分析以下金融资讯：\n{news_text}"},
        {"role": "assistant", "content": assistant_json_str}
    ]
    
    return {"messages": messages}

def main():
    print("正在从 HuggingFace 拉取 FinGPT 公开数据集 (可能需要科学上网)...")
    # 这里使用 FinGPT 的开源指令微调数据集作为示例
    dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    
    print(f"成功拉取数据，总计 {len(dataset)} 条。正在转换为 Qwen ChatML 格式...")
    
    # 格式转换
    chatml_dataset = dataset.map(convert_to_chatml, remove_columns=dataset.column_names)
    
    # 存入本地 JSONL 文件
    output_file = "qwen_finance_sft.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in chatml_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ SFT 训练集转换完成！已保存至 {output_file}")
    
    # 打印一条出来看看效果
    print("\n【数据样例预览】:")
    print(json.dumps(chatml_dataset[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()