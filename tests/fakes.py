from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """Replies with the scripted messages in order; fails the test if asked once more.

    Like the Anthropic API, rejects tool_use/tool_result history when no tools are bound.
    """

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self.bind(tools=[getattr(t, "name", t) for t in tools])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if not kwargs.get("tools") and any(
            isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls) for m in messages
        ):
            raise AssertionError("tool_use/tool_result messages sent without tools bound")
        if not self.responses:
            raise AssertionError("ScriptedChatModel ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name: str, args: dict, id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])
