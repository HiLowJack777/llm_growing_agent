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

if __name__ == '__main__':
    unittest.main()
