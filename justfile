rho-gemma:
    OPENAI_BASE_URL="http://127.0.0.1:8001/v1" \
    OPENAI_API_MODE="chat_completions" \
    uv run rho --provider openai --model gemma-4-E4B-it

llm:
    /home/nico/Projects/llama.cpp/build/bin/llama-server \
        --model /home/nico/Projects/unsloth/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-UD-Q4_K_XL.gguf \
        --temp 1.0 \
        --top-p 0.95 \
        --top-k 64 \
        --alias "unsloth/gemma-4-E4B-it-GGUF" \
        --port 8001 \
        --chat-template-kwargs '{"enable_thinking":true}'
