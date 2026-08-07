# 大模型简洁回答模块

复制环境变量示例并填写 LiteLLM 地址、密钥和模型：

```powershell
Copy-Item llm_inference/.env.example llm_inference/.env
```

直接提问：

```powershell
python -m llm_inference "为什么天空是蓝色的？"
```

命令行固定输出标准 JSON：

```json
{"answer":"因为空气分子更容易散射阳光中的蓝光，所以天空看起来是蓝色的。"}
```

不传问题时可以交互输入：

```powershell
python -m llm_inference
```

查看代理支持的模型或临时选择模型：

```powershell
python -m llm_inference --list-models
python -m llm_inference --model qwen-flash "用一句话解释人工智能"
```

在其他 Python 代码中获取回答文本：

```python
from llm_inference import answer_question, answer_question_text

result = answer_question("太阳为什么会发光？")
answer_text = result["answer"]
print(answer_text)

# 直接交给 TTS 时也可以使用纯文本快捷接口
answer_text = answer_question_text("太阳为什么会发光？")
```

需要连续调用时可以复用客户端，避免每次重新创建连接配置：

```python
from llm_inference import LiteLLMClient

client = LiteLLMClient()
result = client.answer("太阳为什么会发光？")  # {"answer": "..."}
tts_text = client.answer_text("太阳为什么会发光？")  # 可直接传给语音合成

# 流式方法只产生 answer 正文，不包含外层 JSON 字符。
for delta in client.stream_answer_text("太阳为什么会发光？"):
    print(delta, end="", flush=True)
```

默认系统提示词位于 `prompts.py`。它要求模型使用通俗语言、先给结论、通常只回答
2 到 4 句话，并输出 `{"answer":"..."}`。客户端还会解析、校验和清理结果，保证
调用方总是得到只有 `answer` 字段的字典；`answer_question_text` 可直接用于 TTS。
