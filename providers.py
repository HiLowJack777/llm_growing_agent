"""Model providers for the thinking agent.

Each provider wraps one vendor SDK behind the same small interface, so the
agent never touches vendor-specific message formats:

    complete(system, prompt, light=False) -> str
    run_with_tools(system, prompt, tools, handle_tool, max_rounds) -> str
    web_search(query) -> dict
    list_models() -> list[str]

Tools are described once, provider-neutrally, as
    {"name": ..., "description": ..., "parameters": <JSON schema>}
and each provider converts them to its own format. ``handle_tool(name, args)``
is supplied by the agent; it receives the tool name and a parsed argument dict
and returns a JSON-serialisable result.

Each provider runs its own tool loop because the three APIs represent tool
calls and results differently. On the last allowed round tools are disabled
so the model has to give a final answer.

Vendor SDKs are imported lazily, so you only need the SDK for the provider
you actually use.
"""

import json
import os


class ProviderError(Exception):
    """Raised when a provider call fails."""


def _dump(result):
    return json.dumps(result, default=str)


class LLMProvider:
    name = "base"
    display_name = "Base"
    env_var = None
    # Default model IDs. None means "pick from the provider's live model list".
    default_model = None
    default_light_model = None

    def __init__(self, api_key=None, model=None, light_model=None, client=None):
        self.api_key = api_key or (os.environ.get(self.env_var) if self.env_var else None)
        self.model = model or self.default_model
        # The light model handles cheap side tasks (scoring prompts).
        self.light_model = light_model or self.default_light_model or self.model
        self.client = client if client is not None else self._make_client()

    def _make_client(self):
        raise NotImplementedError

    def _model_for(self, light):
        return self.light_model if light else self.model

    def list_models(self):
        raise NotImplementedError

    def complete(self, system, prompt, light=False):
        raise NotImplementedError

    def run_with_tools(self, system, prompt, tools, handle_tool, max_rounds=5):
        raise NotImplementedError

    def web_search(self, query):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    name = "anthropic"
    display_name = "Anthropic (Claude)"
    env_var = "ANTHROPIC_API_KEY"
    default_model = "claude-opus-5-5"
    default_light_model = "claude-haiku-5-5"

    # Models that accept server-side refusal fallbacks: if the model declines a
    # request, the API re-runs it on a fallback model inside the same call.
    FALLBACK_MODELS = {
        "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5",
        "claude-opus-5", "claude-sonnet-5-5",
    }
    FALLBACK_BETA = "server-side-fallback-2026-07-01"
    WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}

    def __init__(self, *args, effort="medium", light_effort="low", **kwargs):
        # Effort controls how much the model thinks (and spends). Claude Opus 5.5
        # always thinks; effort is the cost/quality dial.
        self.effort = effort
        self.light_effort = light_effort
        super().__init__(*args, **kwargs)

    def _make_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()

    def list_models(self):
        try:
            return [m.id for m in self.client.models.list()]
        except Exception as e:
            raise ProviderError(str(e)) from e

    def _create(self, model, system, messages, effort=None, **kwargs):
        params = dict(model=model, max_tokens=16000, system=system, messages=messages, **kwargs)
        if effort:
            params["output_config"] = {"effort": effort}
        try:
            if model in self.FALLBACK_MODELS:
                return self.client.beta.messages.create(
                    betas=[self.FALLBACK_BETA], fallbacks="default", **params
                )
            return self.client.messages.create(**params)
        except Exception as e:
            raise ProviderError(str(e)) from e

    @staticmethod
    def _text(response):
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return f"[Model declined this request{f' ({category})' if category else ''}]"
        return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")

    def complete(self, system, prompt, light=False):
        response = self._create(
            self._model_for(light), system,
            [{"role": "user", "content": prompt}],
            effort=self.light_effort if light else self.effort,
        )
        return self._text(response)

    def run_with_tools(self, system, prompt, tools, handle_tool, max_rounds=5):
        claude_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        messages = [{"role": "user", "content": prompt}]
        response = None
        for round_number in range(max_rounds + 1):
            last_round = round_number == max_rounds
            response = self._create(
                self.model, system, messages, effort=self.effort,
                tools=claude_tools,
                # Tools stay defined (the history holds tool_use blocks), but on
                # the last round the model may not call them.
                **({"tool_choice": {"type": "none"}} if last_round else {}),
            )
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break
            # Pass the full content back unchanged (it carries thinking blocks),
            # then every tool result in one user message.
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_uses:
                result = handle_tool(block.name, dict(block.input or {}))
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _dump(result),
                    **({"is_error": True} if isinstance(result, dict) and result.get("success") is False else {}),
                })
            messages.append({"role": "user", "content": results})
        return self._text(response) if response is not None else ""

    def web_search(self, query):
        messages = [{"role": "user", "content": query}]
        try:
            # Server-side tool: Anthropic runs the search. A long search may
            # pause; resend to let it continue.
            for _ in range(3):
                response = self._create(
                    self.model, "Search the web and summarise what you find, with sources.",
                    messages, effort=self.light_effort, tools=[self.WEB_SEARCH_TOOL],
                )
                if response.stop_reason != "pause_turn":
                    break
                messages = [messages[0], {"role": "assistant", "content": response.content}]
            return {"success": True, "results": self._text(response)}
        except ProviderError as e:
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    name = "openai"
    display_name = "OpenAI"
    env_var = "OPENAI_API_KEY"
    # No hard-coded default: the model is chosen from the live model list at
    # setup and saved per agent, so a renamed or retired model can't break it.
    default_model = None
    default_light_model = None

    _EXCLUDE = ("audio", "realtime", "tts", "transcribe", "image", "embedding",
                "moderation", "search", "whisper", "dall-e", "davinci", "babbage", "instruct")

    def _make_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key) if self.api_key else OpenAI()

    def list_models(self):
        try:
            ids = [m.id for m in self.client.models.list()]
        except Exception as e:
            raise ProviderError(str(e)) from e
        chat = [i for i in ids if (i.startswith("gpt-") or i[:2] in ("o1", "o3", "o4", "o5"))
                and not any(x in i for x in self._EXCLUDE)]
        return sorted(chat) or sorted(ids)

    def _create(self, **params):
        try:
            return self.client.chat.completions.create(**params)
        except Exception as e:
            raise ProviderError(str(e)) from e

    def complete(self, system, prompt, light=False):
        response = self._create(
            model=self._model_for(light),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def run_with_tools(self, system, prompt, tools, handle_tool, max_rounds=5):
        openai_tools = [{"type": "function", "function": t} for t in tools]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        message = None
        for round_number in range(max_rounds + 1):
            params = dict(model=self.model, messages=messages)
            if round_number < max_rounds:
                params.update(tools=openai_tools, tool_choice="auto")
            message = self._create(**params).choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                break
            # One assistant message per round, then one tool message per call.
            messages.append(message.model_dump(exclude_none=True))
            for call in tool_calls:
                try:
                    args = json.loads(call.function.arguments) if call.function.arguments else {}
                except Exception:
                    args = {}
                result = handle_tool(call.function.name, args)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _dump(result)})
        return (message.content or "") if message is not None else ""

    def web_search(self, query):
        try:
            response = self.client.responses.create(
                model=self.model, input=query, tools=[{"type": "web_search"}]
            )
            return {"success": True, "results": response.output_text}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    name = "gemini"
    display_name = "Google Gemini"
    env_var = "GEMINI_API_KEY"
    # Chosen from the live model list at setup and saved per agent.
    default_model = None
    default_light_model = None

    def __init__(self, *args, **kwargs):
        if not kwargs.get("api_key") and not os.environ.get(self.env_var):
            kwargs["api_key"] = os.environ.get("GOOGLE_API_KEY")
        super().__init__(*args, **kwargs)

    def _make_client(self):
        from google import genai
        return genai.Client(api_key=self.api_key) if self.api_key else genai.Client()

    def list_models(self):
        try:
            names = []
            for m in self.client.models.list():
                if "generateContent" in (getattr(m, "supported_actions", None) or []):
                    names.append(m.name.removeprefix("models/"))
        except Exception as e:
            raise ProviderError(str(e)) from e
        return sorted(n for n in names if n.startswith("gemini")) or sorted(names)

    def _generate(self, model, contents, **config):
        from google.genai import types
        try:
            return self.client.models.generate_content(
                model=model, contents=contents, config=types.GenerateContentConfig(**config)
            )
        except Exception as e:
            raise ProviderError(str(e)) from e

    @staticmethod
    def _text(response):
        candidates = getattr(response, "candidates", None) or []
        if not candidates or not getattr(candidates[0], "content", None):
            return ""
        return "".join(p.text for p in (candidates[0].content.parts or []) if getattr(p, "text", None))

    def complete(self, system, prompt, light=False):
        response = self._generate(self._model_for(light), prompt, system_instruction=system)
        return self._text(response)

    def run_with_tools(self, system, prompt, tools, handle_tool, max_rounds=5):
        from google.genai import types
        gemini_tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
            ) for t in tools
        ])]
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        response = None
        for round_number in range(max_rounds + 1):
            mode = "NONE" if round_number == max_rounds else "AUTO"
            response = self._generate(
                self.model, contents,
                system_instruction=system,
                tools=gemini_tools,
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode)),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            calls = getattr(response, "function_calls", None) or []
            if not calls:
                break
            # Keep the model's turn as-is, then answer every call in one turn.
            contents.append(response.candidates[0].content)
            parts = []
            for call in calls:
                result = handle_tool(call.name, dict(call.args or {}))
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=getattr(call, "id", None), name=call.name, response={"result": json.loads(_dump(result))}
                )))
            contents.append(types.Content(role="user", parts=parts))
        return self._text(response) if response is not None else ""

    def web_search(self, query):
        from google.genai import types
        try:
            response = self._generate(
                self.model, query, tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            return {"success": True, "results": self._text(response)}
        except ProviderError as e:
            return {"success": False, "error": str(e)}


PROVIDERS = {p.name: p for p in (AnthropicProvider, OpenAIProvider, GeminiProvider)}


def create_provider(name, **kwargs):
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ProviderError(f"Unknown provider: {name}. Choose one of: {', '.join(PROVIDERS)}")
    return cls(**kwargs)
