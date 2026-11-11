import os
import sqlite3
import tempfile
import unittest

from llm_life_agent import TOOLS, ThinkingAgent
from providers import ProviderError


class FakeProvider:
    """Stands in for a real provider. Scripted tool calls run through the
    agent's real tool handler, so the agent logic is tested end to end."""
    display_name = "Fake"

    def __init__(self, reply="Test response", tool_script=None, final="Final answer"):
        self.reply = reply
        self.tool_script = tool_script or []  # list of rounds; each a list of (name, args)
        self.final = final
        self.complete_calls = []
        self.tool_calls = []
        self.tool_results = []

    def complete(self, system, prompt, light=False):
        self.complete_calls.append({"system": system, "prompt": prompt, "light": light})
        return self.reply

    def run_with_tools(self, system, prompt, tools, handle_tool, max_rounds=5):
        self.tool_calls.append({"system": system, "prompt": prompt, "tools": tools, "max_rounds": max_rounds})
        for round_calls in self.tool_script[:max_rounds]:
            for name, args in round_calls:
                self.tool_results.append(handle_tool(name, args))
        return self.final

    def web_search(self, query):
        return {"success": True, "results": f"results for {query}"}


class FailingProvider(FakeProvider):
    def complete(self, *args, **kwargs):
        raise ProviderError("boom")

    def run_with_tools(self, *args, **kwargs):
        raise ProviderError("boom")


class ThinkingAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        self.provider = FakeProvider()
        self.agent = ThinkingAgent(self.provider, db_path=self.temp_db.name)

    def tearDown(self):
        os.unlink(self.temp_db.name)

    def _configure_identity(self, purpose="Learn Spanish"):
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("INSERT INTO agent_identity (purpose, personality) VALUES (?, 'Patient')", (purpose,))
        conn.commit()
        conn.close()
        self.assertTrue(self.agent.load_active_identity())

    # --- database ------------------------------------------------------
    def test_database_initialization(self):
        conn = sqlite3.connect(self.temp_db.name)
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        for table in ('thoughts', 'insights', 'schema_history', 'agent_identity', 'agent_settings'):
            self.assertIn(table, tables)

    def test_execute_and_query_sql(self):
        result = self.agent.execute_sql("INSERT INTO thoughts (content) VALUES ('Hello')")
        self.assertTrue(result['success'])
        query = self.agent.query_database("SELECT content FROM thoughts")
        self.assertEqual(query['rows'][0][0], 'Hello')

    def test_short_term_memory_limit(self):
        for i in range(25):
            self.agent.update_short_term_memory('speaker', f'message {i}')
        self.assertEqual(len(self.agent.short_term_memory), 20)
        self.assertEqual(self.agent.short_term_memory[0]['message'], 'message 5')

    def test_get_schema_and_latest_schema_text(self):
        self.agent.execute_sql("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
        schema = self.agent.get_schema()
        self.assertTrue(any('test_table' in s for s in schema['schema']))
        self.assertIn('test_table', self.agent.get_latest_schema_text())

    def test_settings_round_trip(self):
        self.assertIsNone(self.agent.get_setting("provider"))
        self.agent.set_setting("provider", "anthropic")
        self.agent.set_setting("provider", "gemini")
        self.assertEqual(self.agent.get_setting("provider"), "gemini")

    # --- identity --------------------------------------------------------
    def test_think_uses_configured_identity(self):
        self._configure_identity("Learn Spanish")
        self.agent.think("Hello")
        system = self.provider.tool_calls[0]["system"]
        self.assertIn("Learn Spanish", system)
        self.assertNotIn("become as interesting as possible", system)

    def test_reflection_and_scoring_use_identity(self):
        self._configure_identity("Learn Spanish")
        self.agent.generate_reflection_prompt()
        self.assertIn("Learn Spanish", self.provider.complete_calls[-1]["system"])

    # --- tools -----------------------------------------------------------
    def test_think_passes_all_tools_and_round_limit(self):
        self.agent.max_tool_rounds = 3
        self.agent.think("Hello")
        call = self.provider.tool_calls[0]
        self.assertEqual({t["name"] for t in call["tools"]}, {t["name"] for t in TOOLS})
        self.assertEqual(call["max_rounds"], 3)

    def test_tool_calls_execute_and_are_reported(self):
        self.agent.execute_sql("INSERT INTO thoughts (content) VALUES ('earlier thought')")
        self.provider.tool_script = [[("query_data", {"sql": "SELECT content FROM thoughts"})]]
        result = self.agent.think("What do you remember?")
        self.assertEqual(self.provider.tool_results[0]["rows"][0][0], "earlier thought")
        self.assertIn("Tool Execution Result (query_data)", result)
        self.assertIn("Final answer", result)
        self.assertTrue(any("query_data" in r for r in self.agent.last_tool_results))

    def test_store_data_skips_forced_thought(self):
        self.provider.tool_script = [[("store_data", {"sql": "INSERT INTO thoughts (content) VALUES ('stored')"})]]
        result = self.agent.think("go")
        self.assertNotIn("Stored response as thought", result)
        rows = self.agent.query_database("SELECT content FROM thoughts")["rows"]
        self.assertEqual([r[0] for r in rows], ["stored"])

    def test_unstored_response_becomes_thought(self):
        result = self.agent.think("go")
        self.assertIn("Stored response as thought", result)
        rows = self.agent.query_database("SELECT content, category FROM thoughts")["rows"]
        self.assertEqual(rows[0], ("Final answer", "reflection"))

    def test_execute_tool_accepts_json_string_args(self):
        result = self.agent.execute_tool("query_data", '{"sql": "SELECT 1"}')
        self.assertEqual(result["rows"][0][0], 1)
        self.assertIn("error", self.agent.execute_tool("nope", {}))

    def test_web_search_goes_through_provider(self):
        self.assertEqual(self.agent.execute_tool("web_search", {"query": "x"})["results"], "results for x")

    def test_provider_failure_is_handled(self):
        agent = ThinkingAgent(FailingProvider(), db_path=self.temp_db.name)
        self.assertEqual(agent.think("hi"), "Failed to process thought")
        self.assertEqual(agent.generate_reflection_prompt(), "Failed to generate reflection prompt")
        self.assertIsNone(agent.evaluate_prompt_effectiveness("p", "r"))

    # --- scores are recorded and used ----------------------------------
    def test_first_evaluation_sets_score_exactly(self):
        prompt = self.agent.generate_reflection_prompt()
        strategy_id = self.agent.current_strategy_id
        self.provider.reply = "Score: 0.8."
        score = self.agent.evaluate_prompt_effectiveness(prompt, "response", strategy_id)
        self.assertEqual(score, 0.8)
        self.assertTrue(self.provider.complete_calls[-1]["light"])  # scoring uses the light model
        conn = sqlite3.connect(self.temp_db.name)
        row = conn.execute("SELECT effectiveness_score, times_used FROM reflection_strategies WHERE id = ?", (strategy_id,)).fetchone()
        conn.close()
        self.assertAlmostEqual(row[0], 0.8)
        self.assertEqual(row[1], 1)

    def test_parse_score_clamps_and_handles_noise(self):
        self.assertEqual(ThinkingAgent.parse_score("0.65"), 0.65)
        self.assertEqual(ThinkingAgent.parse_score("I'd say 1.5"), 1.0)
        self.assertIsNone(ThinkingAgent.parse_score("no idea"))
        self.assertIsNone(ThinkingAgent.parse_score(None))

    def test_reflection_prompt_includes_best_and_worst_strategies(self):
        conn = sqlite3.connect(self.temp_db.name)
        conn.executemany(
            "INSERT INTO reflection_strategies (prompt_template, effectiveness_score, times_used) VALUES (?, ?, ?)",
            [("GREAT PROMPT", 0.9, 3), ("WEAK PROMPT", 0.1, 2), ("UNSCORED PROMPT", None, 0)],
        )
        conn.commit()
        conn.close()
        self.agent.generate_reflection_prompt()
        content = self.provider.complete_calls[-1]["prompt"]
        self.assertIn("GREAT PROMPT", content)
        self.assertIn("WEAK PROMPT", content)
        self.assertNotIn("UNSCORED PROMPT", content)
        self.assertLess(content.index("GREAT PROMPT"), content.index("WEAK PROMPT"))


if __name__ == '__main__':
    unittest.main()
