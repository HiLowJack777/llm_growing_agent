"""Provider adapters, tested against mocked SDK clients (no network)."""
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

from providers import (AnthropicProvider, GeminiProvider, OpenAIProvider,
                       ProviderError, create_provider)

TOOLS = [{
    "name": "query_data",
    "description": "Query the database",
    "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
}]


def recorder(result=None):
    calls = []

    def handle(name, args):
        calls.append((name, args))
        return result if result is not None else {"success": True, "rows": [[1]]}
    return handle, calls


class AnthropicProviderTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.provider = AnthropicProvider(api_key="k", client=self.client)

    @staticmethod
    def _response(blocks, stop_reason="end_turn"):
        return NS(content=blocks, stop_reason=stop_reason, stop_details=None)

    def test_defaults(self):
        self.assertEqual(self.provider.model, "claude-opus-5-5")
        self.assertEqual(self.provider.light_model, "claude-haiku-5-5")

    def test_complete_main_model_uses_fallbacks_and_effort(self):
        self.client.beta.messages.create.return_value = self._response([NS(type="text", text="hi")])
        self.assertEqual(self.provider.complete("sys", "hello"), "hi")
        kwargs = self.client.beta.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-opus-5-5")
        self.assertEqual(kwargs["fallbacks"], "default")
        self.assertEqual(kwargs["output_config"], {"effort": "medium"})
        self.assertEqual(kwargs["system"], "sys")

    def test_complete_light_model_uses_standard_endpoint(self):
        self.client.messages.create.return_value = self._response([NS(type="text", text="0.7")])
        self.assertEqual(self.provider.complete("sys", "score", light=True), "0.7")
        kwargs = self.client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-haiku-5-5")
        self.assertEqual(kwargs["output_config"], {"effort": "low"})
        self.assertNotIn("fallbacks", kwargs)

    def test_tool_loop_returns_results_in_one_user_message(self):
        thinking = NS(type="thinking", thinking="")
        use_a = NS(type="tool_use", id="a", name="query_data", input={"sql": "SELECT 1"})
        use_b = NS(type="tool_use", id="b", name="query_data", input={"sql": "SELECT 2"})
        first = self._response([thinking, use_a, use_b], stop_reason="tool_use")
        final = self._response([NS(type="text", text="done")])
        self.client.beta.messages.create.side_effect = [first, final]
        handle, calls = recorder()

        result = self.provider.run_with_tools("sys", "go", TOOLS, handle)

        self.assertEqual(result, "done")
        self.assertEqual(calls, [("query_data", {"sql": "SELECT 1"}), ("query_data", {"sql": "SELECT 2"})])
        messages = self.client.beta.messages.create.call_args_list[1].kwargs["messages"]
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIs(messages[1]["content"], first.content)  # thinking blocks passed back unchanged
        self.assertEqual([r["tool_use_id"] for r in messages[2]["content"]], ["a", "b"])
        tools = self.client.beta.messages.create.call_args.kwargs["tools"]
        self.assertEqual(tools[0]["input_schema"], TOOLS[0]["parameters"])

    def test_last_round_disables_tool_calls(self):
        use = NS(type="tool_use", id="a", name="query_data", input={"sql": "x"})
        looping = self._response([use], stop_reason="tool_use")
        self.client.beta.messages.create.side_effect = [looping, looping, self._response([NS(type="text", text="end")])]
        handle, _ = recorder()
        self.assertEqual(self.provider.run_with_tools("sys", "go", TOOLS, handle, max_rounds=2), "end")
        calls = self.client.beta.messages.create.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[-1].kwargs["tool_choice"], {"type": "none"})
        self.assertNotIn("tool_choice", calls[0].kwargs)

    def test_refusal_is_reported(self):
        response = NS(content=[], stop_reason="refusal", stop_details=NS(category="cyber"))
        self.client.beta.messages.create.return_value = response
        self.assertIn("declined", self.provider.complete("sys", "x"))

    def test_api_errors_become_provider_errors(self):
        self.client.beta.messages.create.side_effect = RuntimeError("bad key")
        with self.assertRaises(ProviderError):
            self.provider.complete("sys", "x")

    def test_web_search_uses_server_tool(self):
        self.client.beta.messages.create.return_value = self._response([NS(type="text", text="news")])
        self.assertEqual(self.provider.web_search("q"), {"success": True, "results": "news"})
        tools = self.client.beta.messages.create.call_args.kwargs["tools"]
        self.assertEqual(tools[0]["type"], "web_search_20260209")


class OpenAIProviderTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.provider = OpenAIProvider(api_key="k", model="main-model", light_model="small-model", client=self.client)

    @staticmethod
    def _reply(content=None, tool_calls=None):
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls
        message.model_dump.return_value = {"role": "assistant", "content": content, "tool_calls": []}
        return MagicMock(choices=[MagicMock(message=message)])

    @staticmethod
    def _call(call_id, args):
        call = MagicMock()
        call.id = call_id
        call.function.name = "query_data"
        call.function.arguments = json.dumps(args)
        return call

    def test_complete_uses_light_model_when_asked(self):
        self.client.chat.completions.create.return_value = self._reply("ok")
        self.provider.complete("sys", "x", light=True)
        self.assertEqual(self.client.chat.completions.create.call_args.kwargs["model"], "small-model")

    def test_tool_results_fed_back_in_same_turn(self):
        first = self._reply(tool_calls=[self._call("a", {"sql": "S1"}), self._call("b", {"sql": "S2"})])
        self.client.chat.completions.create.side_effect = [first, self._reply("done")]
        handle, calls = recorder()
        self.assertEqual(self.provider.run_with_tools("sys", "go", TOOLS, handle), "done")
        self.assertEqual([c[1]["sql"] for c in calls], ["S1", "S2"])
        messages = self.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "tool"])

    def test_last_round_withholds_tools(self):
        looping = self._reply(tool_calls=[self._call("c", {"sql": "x"})])
        self.client.chat.completions.create.side_effect = [looping, looping, self._reply("final")]
        handle, _ = recorder()
        self.assertEqual(self.provider.run_with_tools("sys", "go", TOOLS, handle, max_rounds=2), "final")
        self.assertNotIn("tools", self.client.chat.completions.create.call_args.kwargs)

    def test_list_models_filters_non_chat_models(self):
        ids = ["gpt-x", "text-embedding-3", "gpt-x-realtime", "o4-mini", "dall-e-3"]
        self.client.models.list.return_value = [NS(id=i) for i in ids]
        self.assertEqual(self.provider.list_models(), ["gpt-x", "o4-mini"])

    def test_no_hardcoded_default_model(self):
        self.assertIsNone(OpenAIProvider.default_model)


class GeminiProviderTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.provider = GeminiProvider(api_key="k", model="gemini-main", light_model="gemini-small", client=self.client)

    @staticmethod
    def _response(text=None, function_calls=None):
        from google.genai import types
        parts = [types.Part.from_text(text=text)] if text else []
        for call in function_calls or []:
            parts.append(types.Part(function_call=call))
        content = types.Content(role="model", parts=parts)
        return NS(candidates=[NS(content=content)], function_calls=function_calls or None)

    def test_tool_loop(self):
        from google.genai import types
        call = types.FunctionCall(id="a", name="query_data", args={"sql": "SELECT 1"})
        self.client.models.generate_content.side_effect = [self._response(function_calls=[call]), self._response("done")]
        handle, calls = recorder()

        self.assertEqual(self.provider.run_with_tools("sys", "go", TOOLS, handle), "done")
        self.assertEqual(calls, [("query_data", {"sql": "SELECT 1"})])
        second = self.client.models.generate_content.call_args_list[1].kwargs
        reply = second["contents"][-1].parts[0].function_response
        self.assertEqual((reply.id, reply.name), ("a", "query_data"))
        self.assertEqual(reply.response["result"]["rows"], [[1]])
        self.assertEqual(second["config"].system_instruction, "sys")

    def test_last_round_sets_function_calling_none(self):
        from google.genai import types
        call = types.FunctionCall(id="a", name="query_data", args={"sql": "x"})
        self.client.models.generate_content.side_effect = [
            self._response(function_calls=[call]), self._response("end")]
        handle, _ = recorder()
        self.provider.run_with_tools("sys", "go", TOOLS, handle, max_rounds=1)
        config = self.client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.tool_config.function_calling_config.mode, "NONE")

    def test_complete_light(self):
        self.client.models.generate_content.return_value = self._response("0.4")
        self.assertEqual(self.provider.complete("sys", "x", light=True), "0.4")
        self.assertEqual(self.client.models.generate_content.call_args.kwargs["model"], "gemini-small")

    def test_list_models_keeps_generate_content_models(self):
        self.client.models.list.return_value = [
            NS(name="models/gemini-a", supported_actions=["generateContent"]),
            NS(name="models/embed-b", supported_actions=["embedContent"]),
        ]
        self.assertEqual(self.provider.list_models(), ["gemini-a"])


class FactoryTests(unittest.TestCase):
    def test_unknown_provider(self):
        with self.assertRaises(ProviderError):
            create_provider("cursor")


if __name__ == "__main__":
    unittest.main()
