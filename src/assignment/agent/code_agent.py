"""The Part 1 coding agent: fix a software issue and submit a git patch.
第 1 部分 编码任务：修复一个软件问题并提交一个 Git 补丁"""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch.
    负责修复软件问题并提交 Git 补丁的agent"""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # √TODO(Part 1.3): Make the `execute` and `send_message` tools available to the agent.
        #让代理能够使用 `execute` 和 `send_message` 工具。
        self.tools.extend([EXECUTE_TOOL,SEND_MESSAGE_TOOL])
        
        # √TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        #构建系统提示和任务提示。这些提示应可被 `Agent.build_prompt` 方法使用
        self.system_prompt=(
            "You are a coding agent. You interact with a Linux sandbox through "
            "tools: use the `execute` tool to run shell commands, inspect the "
            "repository, fix the issue, and verify your fix. Think step by step, "
            "act one command at a time, and submit a git patch when done.\n\n"
            f"<system_information>\n"
            f"{{\n"
            f'  "machine": "{self.env.machine}",\n'
            f'  "release": "{self.env.release}",\n'
            f'  "system": "{self.env.system}",\n'
            f'  "version": "{self.env.version}"\n'
            f"}}\n"
            f"</system_information>"
        )
        self.task_prompt=self.task
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        #√TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        #解析每个调用，执行识别出的工具，并为每个调用返回一条消息（一个代理响应中可能包含多个工具调用！）。
        #格式错误的 JSON 和未知工具必须作为可恢复的观察结果传递给代理，而非作为异常处理。
        observations:list[dict[str,str]] = []

        for call in tool_calls:
            call_id = call.get("id","")
            function = call.get("function",{})
            name = function.get("name","")
            raw_arguments = function.get("arguments","{}")
        
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments,dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError,ValueError) as exc:
                observations.append({
                    "role":"tool",
                    "tool_call_id":call_id,
                    "content": f"Error:invalid tool arguments:{exc}" ,
                })
                continue
            
            if name=="execute":
                result=self.env.execute(
                    arguments["command"],
                    timeout=arguments.get("timeout"),
                    cwd=arguments.get("cwd"),
                    env=arguments.get("env"),
                    shell=arguments.get("shell")
                )
                content = format_tool_output(result)
            
            elif name=="send_message":
                content=f"Message sent to user:{arguments.get("summery","")}"
            
            else:
                content=f"Error: unknown tool '{name}'."

            observations.append(
                {"role":"tool",
                "tool_call_id":call_id,
                "content":content}
            )

        return observations
