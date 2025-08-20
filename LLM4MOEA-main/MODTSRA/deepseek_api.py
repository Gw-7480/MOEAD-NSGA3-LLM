# deepseek_api.py
import os
from openai import OpenAI

# 初始化 client
client = OpenAIclient = OpenAI(
    api_key="sk-7e41387f5f5e4d4bb4788071a92cb224",
    base_url="https://api.deepseek.com"
)

def deepseek_call(prompt: str,
                  model: str = "deepseek-chat",
                  temperature: float = 0.2,
                  max_tokens: int = 1500) -> str:
    """
    调用 DeepSeek API，返回 LLM 的原始输出字符串
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content