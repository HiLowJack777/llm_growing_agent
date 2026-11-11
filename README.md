# LLM Growing Agent — A Self-Evolving AI

This project is a **self-reflective AI agent** that thinks, stores its thoughts, evolves its own memory schema, and is interactable in a simple desktop GUI.

It’s designed to be a complete thinking agent with journaling, reflection prompts, SQL schema evolution, and web search capabilities.

## Features
- Autonomous journaling and reflection
- Self-evolving memory using SQLite
- Schema awareness and auto-migration
- Short-term memory buffer
- Reflection prompt scoring and strategy evaluation
- Choice of engine: Anthropic (Claude), OpenAI, or Google Gemini, saved per agent
- Web search through the chosen engine's built-in search tool
- Simple Tkinter-based GUI interface

## Start

1. **Clone the repo and install dependencies**
   ```bash
   git clone https://github.com/your-username/llm_growing_agent.git
   cd llm_growing_agent
   pip install -r requirements.txt
   ```
2. **Set an API key** for the engine you want (or paste it when asked):
   | Engine | Environment variable |
   |---|---|
   | Anthropic (Claude) | `ANTHROPIC_API_KEY` |
   | OpenAI | `OPENAI_API_KEY` |
   | Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
3. **Run it and create your first agent:** `python llm_life_agent.py`

## Engines and models

Each agent remembers its engine and models in its own `.db` file, so different
agents can run on different engines. Each agent uses two models:

- **Main model:** thinking, tool use and writing reflection prompts.
- **Light model:** cheap side tasks (currently scoring reflection prompts).

| Engine | Main model | Light model |
|---|---|---|
| Anthropic | `claude-opus-5-5` at `medium` effort | `claude-haiku-5-5` at `low` effort |
| OpenAI | picked from your account's model list at setup | picked at setup |
| Gemini | picked from your account's model list at setup | picked at setup |

For Claude, effort is the main cost dial: lower effort means less thinking and
fewer tokens. To change it, pass `effort=` / `light_effort=` to
`AnthropicProvider`. `claude-sonnet-5-5` costs about half as much as Opus 5.5
and can be picked at setup.

Adding another engine means writing one class in `providers.py` with
`complete`, `run_with_tools`, `web_search` and `list_models`.

See [ROADMAP.md](ROADMAP.md) for known issues and planned work.

## Testing

Run the unit tests with:
```bash
python -m unittest
```
