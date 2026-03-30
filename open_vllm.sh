python -m vllm.entrypoints.openai.api_server \
    --model ./qwen_model \
    --enable-lora \
    --lora-modules finance_news=./saves/qwen2.5-7b-finance-lora \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes