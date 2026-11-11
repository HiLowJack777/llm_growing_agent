import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock

from llm_life_agent import ThinkingAgent

class ThinkingAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.mock_client = MagicMock()
        # Prepare fake response for OpenAI call
        fake_response = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "Test response"
        fake_response.choices = [fake_choice]
        self.mock_client.chat.completions.create.return_value = fake_response
        self.agent = ThinkingAgent(self.mock_client, db_path=self.temp_db.name)

    def tearDown(self):
        os.unlink(self.temp_db.name)

    def test_database_initialization(self):
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        self.assertIn('thoughts', tables)
        self.assertIn('insights', tables)
        self.assertIn('schema_history', tables)
        self.assertIn('agent_identity', tables)

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
        schema_text = self.agent.get_latest_schema_text()
        self.assertIn('test_table', schema_text)

    def test_make_openai_call(self):
        messages = [{'role': 'user', 'content': 'hi'}]
        response = self.agent.make_openai_call(messages)
        self.mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(response.choices[0].message.content, 'Test response')

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _reply(content=None, tool_calls=None):
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls
        message.model_dump.return_value = {"role": "assistant", "content": content, "tool_calls": []}
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    @staticmethod
    def _tool_call(call_id, name, arguments):
        call = MagicMock()
        call.id = call_id
        call.function.name = name
        call.function.arguments = json.dumps(arguments)
        return call

    def _configure_identity(self, purpose="Learn Spanish"):
        conn = sqlite3.connect(self.temp_db.name)
        conn.execute("INSERT INTO agent_identity (purpose, personality) VALUES (?, 'Patient')", (purpose,))
        conn.commit()
        conn.close()
        self.assertTrue(self.agent.load_active_identity())

    # --- think() uses the configured identity --------------------------
    def test_think_uses_configured_identity(self):
        self._configure_identity("Learn Spanish")
        self.agent.think("Hello")
        messages = self.mock_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Learn Spanish", messages[0]["content"])
        self.assertNotIn("become as interesting as possible", messages[0]["content"])

    # --- tool results come back within the same turn -------------------
    def test_tool_results_fed_back_in_same_turn(self):
        self.agent.execute_sql("INSERT INTO thoughts (content) VALUES ('earlier thought')")
        tool_round = self._reply(tool_calls=[
            self._tool_call("call_1", "query_data", {"sql": "SELECT content FROM thoughts"})
        ])
        final = self._reply(content="I recall an earlier thought.")
        self.mock_client.chat.completions.create.side_effect = [tool_round, final]

        result = self.agent.think("What do you remember?")

        self.assertEqual(self.mock_client.chat.completions.create.call_count, 2)
        second_messages = self.mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["tool_call_id"], "call_1")
        self.assertIn("earlier thought", tool_messages[0]["content"])
        # Assistant tool-call message appears exactly once before the tool result
        self.assertEqual(sum(1 for m in second_messages if m.get("role") == "assistant"), 1)
        self.assertIn("I recall an earlier thought.", result)

    def test_multiple_tool_calls_share_one_assistant_message(self):
        tool_round = self._reply(tool_calls=[
            self._tool_call("a", "get_schema", {}),
            self._tool_call("b", "store_data", {"sql": "INSERT INTO thoughts (content) VALUES ('x')"}),
        ])
        self.mock_client.chat.completions.create.side_effect = [tool_round, self._reply(content="done")]
        self.agent.think("go")
        second_messages = self.mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        roles = [m["role"] for m in second_messages]
        self.assertEqual(roles[-3:], ["assistant", "tool", "tool"])

    def test_tool_loop_is_bounded(self):
        self.agent.max_tool_rounds = 2
        looping = self._reply(tool_calls=[self._tool_call("c", "get_schema", {})])
        self.mock_client.chat.completions.create.side_effect = [looping, looping, self._reply(content="final")]
        result = self.agent.think("loop")
        calls = self.mock_client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 3)
        # Final round withholds tools so the model has to answer
        self.assertNotIn("tools", calls[-1].kwargs)
        self.assertIn("final", result)

    # --- scores are recorded and used ----------------------------------
    def test_first_evaluation_sets_score_exactly(self):
        prompt = self.agent.generate_reflection_prompt()
        strategy_id = self.agent.current_strategy_id
        self.mock_client.chat.completions.create.return_value = self._reply(content="Score: 0.8.")
        score = self.agent.evaluate_prompt_effectiveness(prompt, "response", strategy_id)
        self.assertEqual(score, 0.8)
        conn = sqlite3.connect(self.temp_db.name)
        row = conn.execute("SELECT effectiveness_score, times_used FROM reflection_strategies WHERE id = ?", (strategy_id,)).fetchone()
        conn.close()
        self.assertAlmostEqual(row[0], 0.8)
        self.assertEqual(row[1], 1)

    def test_parse_score_clamps_and_handles_noise(self):
        self.assertEqual(ThinkingAgent.parse_score("0.65"), 0.65)
        self.assertEqual(ThinkingAgent.parse_score("I'd say 1.5"), 1.0)
        self.assertIsNone(ThinkingAgent.parse_score("no idea"))

    def test_reflection_prompt_includes_best_and_worst_strategies(self):
        conn = sqlite3.connect(self.temp_db.name)
        conn.executemany(
            "INSERT INTO reflection_strategies (prompt_template, effectiveness_score, times_used) VALUES (?, ?, ?)",
            [("GREAT PROMPT", 0.9, 3), ("WEAK PROMPT", 0.1, 2), ("UNSCORED PROMPT", None, 0)],
        )
        conn.commit()
        conn.close()
        self.agent.generate_reflection_prompt()
        content = self.mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("GREAT PROMPT", content)
        self.assertIn("WEAK PROMPT", content)
        self.assertNotIn("UNSCORED PROMPT", content)
        self.assertLess(content.index("GREAT PROMPT"), content.index("WEAK PROMPT"))

if __name__ == '__main__':
    unittest.main()
