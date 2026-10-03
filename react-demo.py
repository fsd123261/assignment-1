import os
import json
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

# ---------- 1. 从项目 .env 加载配置 ----------
# load_dotenv() 默认从“当前工作目录”向上找 .env；
# 为了无论从哪里运行都能找到，显式指向项目根目录的 .env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")          # 不填则 OpenAI SDK 会用官方地址
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")  # 给个兜底默认值

# 快速失败：缺配置就直接报错，别等到调 API 才发现
if not API_KEY:
    raise RuntimeError("未找到 OPENAI_API_KEY，请检查项目根目录的 .env 文件")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)  # base_url 为 None 时 SDK 用默认值

# ---------- 2. 工具定义 ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the final answer and finish the task.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    },
]

class TaskDone(Exception):
    def __init__(self, answer):
        self.answer = answer

# ---------- 3. 工具执行 ----------
def execute_tool(name: str, args: dict) -> str:
    if name == "bash":
        result = subprocess.run(
            args["command"], shell=True,
            capture_output=True, text=True, timeout=30,
        )
        return (result.stdout + result.stderr)[:2000]
    elif name == "submit":
        raise TaskDone(args["answer"])
    return f"Unknown tool: {name}"

# ---------- 4. ReAct 主循环 ----------
def run_agent(task: str, max_steps: int = 20) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful agent. Use tools step by step to solve the task. Then call `submit` with the final answer."},
        {"role": "user", "content": f"Task: {task}"},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,          # ← 使用 .env 里的模型名
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            print(f"[step {step}] {tc.function.name}({tc.function.arguments})")
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}  # 坏参数：交由 execute_tool 返回错误信息
            try:
                result = execute_tool(tc.function.name, args)
            except TaskDone as e:
                return e.answer
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Exceeded max steps"

if __name__ == "__main__":
    print(f"MODEL  = {MODEL!r}")       # ← 加这行，repr 能暴露隐藏的空格/#/换行
    print(f"BASE   = {os.getenv('OPENAI_BASE_URL')!r}")
    print(f"使用模型: {MODEL} @ {BASE_URL or 'https://api.openai.com/v1'}\n")
    answer = run_agent("当前目录下有哪些 .py 文件？统计总行数，告诉我结果。")
    print("\n最终答案:", answer)
