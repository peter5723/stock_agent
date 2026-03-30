import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ==========================================
# 1. 基础配置
# ==========================================
# 替换为你本地 Qwen2.5-7B 的真实路径
model_path = "./qwen_model" 
data_path = "qwen_finance_sft.jsonl"
output_dir = "./saves/qwen2.5-7b-finance-lora"

print("-> 正在加载 Tokenizer 和 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# Qwen 的 pad_token 通常是 eos_token
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # 如果是 30/40系或 A100，使用 bfloat16 加速且省显存
    device_map="auto",
    trust_remote_code=True
)

# ==========================================
# 2. 配置 LoRA (注入金融语感的关键)
# ==========================================
print("-> 正在注入 LoRA 适配器...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 3. 数据处理 (ChatML 格式套用)
# ==========================================
# ==========================================
# 3. 数据处理 (直接手动 Tokenize，彻底摆脱黑盒框架)
# ==========================================
print("-> 正在准备数据集...")
dataset = load_dataset("json", data_files=data_path, split="train")

def format_and_tokenize(example):
    # example["messages"] 在 batched=True 时是一个批次的列表
    # 我们用原生方法把 ChatML 格式转为纯文本
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in example["messages"]]
    # 直接在这一步完成 Tokenize 和 1024 长度截断
    return tokenizer(texts, truncation=True, max_length=1024)

# 使用 batched=True 开启多进程加速处理
# 必须 remove_columns，清除无用的原始文本字段，否则传入 Trainer 会报错
formatted_dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)

# 原生 Causal LM 数据整理器 (mlm=False 会自动把 labels 设为 input_ids)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================
# 4. 训练参数设置与启动 (使用最稳健的官方 Trainer)
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=200,
    bf16=True, 
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none" # 关掉默认的 wandb 记录，防止未登录报错
)

# 直接使用 Transformers 原生 Trainer！
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    data_collator=collator,
)

print("-> 开始训练！🚀")
trainer.train()

print(f"-> 训练完成，LoRA 权重已保存至：{output_dir}")
trainer.save_model(output_dir)

print("-> 开始训练！🚀")
trainer.train()

print(f"-> 训练完成，LoRA 权重已保存至：{output_dir}")
trainer.save_model(output_dir)