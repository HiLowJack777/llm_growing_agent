import sqlite3
from datetime import datetime
import os
import json
import re
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import tkinter as tk
import threading
from getpass import getpass

from providers import PROVIDERS, ProviderError, create_provider

console = Console()

# Tools the agent can call, described once for every provider.
TOOLS = [
    {
        "name": "execute_migration",
        "description": "Execute a SQL migration to modify the database schema",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL migration command"}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "store_data",
        "description": "Store any type of data in the database. Use this to record any structured data in your tables.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL INSERT command for any table. Examples:\n"
                                   "- INSERT INTO opportunities (name, potential_value) VALUES ('New Market', 50000)\n"
                                   "- INSERT INTO metrics (category, value) VALUES ('ROI', 12.5)\n"
                                   "- INSERT INTO thoughts (content) VALUES ('Strategic insight about...')",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_schema",
        "description": "Get the current database schema",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_data",
        "description": "Query any data from the database tables using SELECT statements",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL SELECT query. Examples:\n"
                                   "- SELECT * FROM thoughts ORDER BY timestamp DESC LIMIT 5\n"
                                   "- SELECT category, COUNT(*) FROM metrics GROUP BY category\n"
                                   "- SELECT * FROM opportunities WHERE potential_value > 1000",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the internet for current information about markets, opportunities, and trends",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to look up information online"}
            },
            "required": ["query"],
        },
    },
]

class ThinkingAgent:
    def __init__(self, provider, db_path="llm_memory.db"):
        self.provider = provider  # an LLMProvider (Claude, OpenAI or Gemini)
        self.db_path = db_path
        self.short_term_memory = []  # STM buffer
        self.system_message = {"role": "system", "content": "You are a self-reflective AI agent with the ability to learn and adapt."}
        self.last_tool_results = []
        self.user_input_queue = None  # Add this to store user input
        self.current_strategy_id = None  # reflection_strategies row for the active prompt
        self.max_tool_rounds = 5  # max tool-call rounds within a single think() turn
        self.initialize_database()
        # Add table for storing reflection strategies
        self.initialize_reflection_table()
        self.initialize_thought_tables()
        self.initialize_agent_identity()
        self.initialize_settings_table()
        # Load existing identity if available
        self.load_active_identity()

    def initialize_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_text TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def initialize_reflection_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reflection_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_template TEXT NOT NULL,
            effectiveness_score FLOAT,
            times_used INTEGER DEFAULT 1,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def initialize_thought_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT,
            confidence FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thought_id INTEGER,
            content TEXT NOT NULL,
            potential_value FLOAT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thought_id) REFERENCES thoughts(id)
        )
        """)
        conn.commit()
        conn.close()

    def initialize_agent_identity(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_identity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            purpose TEXT NOT NULL,
            personality TEXT,
            goals TEXT,
            constraints TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
        """)
        conn.commit()
        conn.close()

    def initialize_settings_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        conn.commit()
        conn.close()

    def get_setting(self, key, default=None):
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT value FROM agent_settings WHERE key = ?", (key,)).fetchone()
            return row[0] if row else default
        finally:
            conn.close()

    def set_setting(self, key, value):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO agent_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
        conn.close()

    def load_active_identity(self):
        """Load the currently active identity configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        SELECT purpose, personality, goals, constraints 
        FROM agent_identity 
        WHERE is_active = 1 
        ORDER BY created_at DESC 
        LIMIT 1
        """)
        identity = cursor.fetchone()
        conn.close()

        if identity:
            purpose, personality, goals, constraints = identity
            self.system_message = {
                "role": "system",
                "content": f"""You are a self-reflective AI agent.
                Primary Purpose: {purpose}
                
                Personality: {personality if personality else 'Analytical and thoughtful'}
                Goals: {goals if goals else 'Continuously learn and improve'}
                Constraints: {constraints if constraints else 'Act ethically and responsibly'}
                
                Always work towards your defined purpose while maintaining your specified characteristics."""
            }
            return True
        return False

    def update_short_term_memory(self, speaker, message):
        self.short_term_memory.append({"speaker": speaker, "message": message})
        if len(self.short_term_memory) > 20:
            self.short_term_memory = self.short_term_memory[-20:]

    def execute_sql(self, sql):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.executescript(sql)
            conn.commit()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def get_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Get table schemas
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
            schema = [row[0] for row in cursor.fetchall()]
            
            # Get index information
            cursor.execute("""
                SELECT 
                    m.tbl_name as table_name,
                    i.name as index_name,
                    GROUP_CONCAT(ii.name) as indexed_columns
                FROM sqlite_master m
                LEFT JOIN pragma_index_list(m.name) i
                LEFT JOIN pragma_index_info(i.name) ii
                WHERE m.type = 'table'
                GROUP BY m.tbl_name, i.name
                HAVING i.name IS NOT NULL
            """)
            indexes = cursor.fetchall()
            
            # Format index information
            index_info = []
            for table, index_name, cols in indexes:
                index_info.append(f"CREATE INDEX {index_name} ON {table} ({cols});")
            
            # Combine schema and index information
            full_schema = schema + index_info
            schema_text = "\n".join(full_schema)
            
            cursor.execute("INSERT INTO schema_history (schema_text) VALUES (?)", (schema_text,))
            conn.commit()
            return {"schema": full_schema}
        finally:
            conn.close()

    def get_latest_schema_text(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT schema_text FROM schema_history ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else "No schema recorded yet."
        finally:
            conn.close()

    def query_database(self, sql):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def perform_web_search(self, query):
        return self.provider.web_search(query)

    def complete(self, prompt, light=False):
        """Single model call with the agent's identity as the system prompt.
        light=True uses the provider's cheaper model. Returns None on failure."""
        try:
            return self.provider.complete(self.system_message["content"], prompt, light=light)
        except ProviderError as e:
            console.print(f"[red]Error calling {self.provider.display_name}: {e}[/red]")
            return None

    def get_scored_strategies(self, best=3, worst=2):
        """Return (best, worst) previously evaluated reflection prompts with their scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            base = """
                SELECT prompt_template, effectiveness_score, times_used
                FROM reflection_strategies
                WHERE effectiveness_score IS NOT NULL AND times_used > 0
            """
            cursor.execute(base + " ORDER BY effectiveness_score DESC, times_used DESC LIMIT ?", (best,))
            top = cursor.fetchall()
            cursor.execute(base + " ORDER BY effectiveness_score ASC, times_used DESC LIMIT ?", (worst,))
            bottom = [row for row in cursor.fetchall() if row not in top]
            return top, bottom
        finally:
            conn.close()

    @staticmethod
    def _format_strategies(rows, max_chars=600):
        if not rows:
            return "None yet."
        lines = []
        for prompt, score, uses in rows:
            text = prompt if len(prompt) <= max_chars else prompt[:max_chars] + "..."
            lines.append(f"[score {score:.2f}, used {uses}x]\n{text}")
        return "\n\n".join(lines)

    def generate_reflection_prompt(self):
        # First, get recent thoughts
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content, timestamp 
            FROM thoughts 
            ORDER BY timestamp DESC 
            LIMIT 3
        """)
        recent_thoughts = cursor.fetchall()
        conn.close()

        recent_thoughts_str = "\n".join([f"- {thought[0]} ({thought[1]})" for thought in recent_thoughts]) if recent_thoughts else "No recent thoughts."
        
        # Format the tool results and memory before using them
        tool_results_str = "\n".join(self.last_tool_results[-3:])  # Last 3 tool results
        memory_str = "\n".join([f"{m['speaker']}: {m['message']}" for m in self.short_term_memory[-5:]])  # Last 5 memories

        # Learn from past evaluations: show what scored well and what scored poorly
        top_strategies, bottom_strategies = self.get_scored_strategies()

        request = f"""
                As a self-reflective agent, generate a detailed reflection prompt for your next thinking cycle.
                Consider:
                1. Your recent activities and their outcomes
                2. Current goals and progress
                3. Areas that need more investigation
                4. Potential new strategies to explore
                
                Recent thoughts:
                {recent_thoughts_str}
                
                Current context:
                - Last tool results: {tool_results_str}
                - Short term memory: {memory_str}
                
                Your highest-scoring past reflection prompts (build on what made these effective):
                {self._format_strategies(top_strategies)}
                
                Your lowest-scoring past reflection prompts (avoid what made these weak):
                {self._format_strategies(bottom_strategies)}
                
                IMPORTANT: Your response MUST include at least one insight or observation that should be stored in the database.
                Include specific questions that will help drive progress toward your primary purpose.
            """

        generated_prompt = self.complete(request)
        if not generated_prompt:
            self.current_strategy_id = None
            return "Failed to generate reflection prompt"

        # Store the generated prompt (unscored and unused until evaluated)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO reflection_strategies (prompt_template, effectiveness_score, times_used)
        VALUES (?, NULL, 0)
        """, (generated_prompt,))
        self.current_strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return generated_prompt

    def think(self, prompt):
        # Add user input to prompt if available
        if self.user_input_queue:
            prompt = f"{prompt}\n\nUser Input: {self.user_input_queue}"
            self.user_input_queue = None  # Clear the queue after using it

        short_term_context = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in self.short_term_memory])

        # Include any previous tool results in the context
        tool_results_context = ""
        if self.last_tool_results:
            tool_results_context = "\nPrevious tool execution results:\n" + "\n".join(self.last_tool_results)

        user_content = f"{prompt}\n\nShort-Term Memory Context:\n{short_term_context}{tool_results_context}"

        # Reset tool results for this iteration
        self.last_tool_results = []
        log = []
        stored = {"any": False}

        def handle_tool(name, args):
            result = self.execute_tool(name, args)
            if name == "store_data" and result.get("success", False):
                stored["any"] = True
            # Keep the result for the next cycle's context, too
            pretty = json.dumps(result, indent=2, default=str)
            self.last_tool_results.append(f"Tool {name}: {pretty}")
            log.append(f"\nTool Execution Result ({name}):\n{pretty}")
            return result

        # The provider runs the tool loop: the model calls tools, sees the
        # results, and continues within this same turn.
        try:
            content = self.provider.run_with_tools(
                self.system_message["content"], user_content, TOOLS, handle_tool,
                max_rounds=self.max_tool_rounds,
            )
        except ProviderError as e:
            console.print(f"[red]Error calling {self.provider.display_name}: {e}[/red]")
            if not log:
                return "Failed to process thought"
            content = None

        formatted_response = "".join(log)

        # If nothing was stored, force a thought storage
        if not stored["any"] and content:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO thoughts (content, category, confidence)
                    VALUES (?, 'reflection', 0.5)
                """, (content[:500],))  # Limit content length
                conn.commit()
                conn.close()
                formatted_response += "\n[System: Stored response as thought]"
            except Exception as e:
                formatted_response += f"\n[System: Failed to store thought: {str(e)}]"

        if content:
            formatted_response += f"\nAssistant Response:\n{content}"

        return formatted_response

    def execute_tool(self, function_name, function_args):
        """Run one tool call from the model and return its result dict."""
        if isinstance(function_args, str):
            try:
                function_args = json.loads(function_args) if function_args else {}
            except Exception:
                function_args = {}
        function_args = function_args or {}

        if function_name in ("execute_migration", "store_data"):
            return self.execute_sql(function_args.get("sql", ""))
        if function_name == "get_schema":
            return self.get_schema()
        if function_name == "query_data":
            return self.query_database(function_args.get("sql", ""))
        if function_name == "web_search":
            return self.perform_web_search(function_args.get("query", ""))
        return {"error": f"Unknown function: {function_name}"}

    def evaluate_prompt_effectiveness(self, prompt, response, strategy_id=None):
        request = f"""
                Evaluate the effectiveness of this reflection prompt and response:
                
                Prompt:
                {prompt}
                
                Response:
                {response}
                
                Rate the effectiveness from 0.0 to 1.0 based on:
                - Depth of reflection
                - Actionable insights generated
                - Progress toward stated goals
                - Strategic thinking demonstrated
                
                Return only the numeric score.
            """

        # Scoring is a simple task, so it runs on the provider's light model.
        score = self.parse_score(self.complete(request, light=True))
        if score is None:
            return None

        # Fold the score into the strategy's running average. A fresh row has
        # times_used = 0 and a NULL score, so the first evaluation sets it outright.
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        update = """
            UPDATE reflection_strategies
            SET effectiveness_score = (COALESCE(effectiveness_score, 0) * times_used + ?) / (times_used + 1),
                times_used = times_used + 1,
                last_used = CURRENT_TIMESTAMP
        """
        if strategy_id is None:
            strategy_id = self.current_strategy_id
        if strategy_id is not None:
            cursor.execute(update + " WHERE id = ?", (score, strategy_id))
        else:
            cursor.execute(update + " WHERE id = (SELECT MAX(id) FROM reflection_strategies WHERE prompt_template = ?)", (score, prompt))
        conn.commit()
        conn.close()
        return score

    @staticmethod
    def parse_score(text):
        """Pull a 0.0-1.0 score out of the model's reply, e.g. '0.8', 'Score: 0.75.'"""
        if not text:
            return None
        match = re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return None
        return max(0.0, min(1.0, float(match.group())))

class SimpleInputGUI:
    def __init__(self, agent):
        self.root = tk.Tk()
        self.root.title("AI Agent Interface")
        self.root.geometry("600x400")  # Made window larger
        self.agent = agent
        
        # Make the window stay on top
        self.root.attributes('-topmost', True)
        
        # Create main frame
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add response display area
        self.response_area = tk.Text(self.frame, height=15, wrap=tk.WORD)
        self.response_area.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.response_area.config(state=tk.DISABLED)  # Make it read-only
        
        # Input area frame
        self.input_frame = tk.Frame(self.frame)
        self.input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Input field
        self.input_field = tk.Text(self.input_frame, height=3)
        self.input_field.pack(fill=tk.X, pady=(0, 5))
        
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.handle_user_input)
        self.send_button.pack(fill=tk.X)
        
        # Bind Enter key to send
        self.input_field.bind('<Control-Return>', lambda e: self.handle_user_input())
        
        # Schedule the agent's background reflection cycle
        self.root.after(10000, self.schedule_reflection)
        
    def display_response(self, response):
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete(1.0, tk.END)
        self.response_area.insert(tk.END, response)
        self.response_area.config(state=tk.DISABLED)
        self.response_area.see(tk.END)
    
    def handle_user_input(self):
        user_input = self.input_field.get("1.0", tk.END).strip()
        if user_input:
            # Clear input field
            self.input_field.delete("1.0", tk.END)
            
            # Disable input while processing
            self.input_field.config(state=tk.DISABLED)
            self.send_button.config(state=tk.DISABLED)
            
            # Process in separate thread to keep GUI responsive
            def process_input():
                self.agent.user_input_queue = user_input
                response = self.agent.think(f"User Input: {user_input}\nPlease respond to this specific input.")
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.display_response(response))
                self.root.after(0, lambda: self.input_field.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
            
            # Start processing thread
            threading.Thread(target=process_input, daemon=True).start()
    
    def schedule_reflection(self):
        def background_reflection():
            reflection_prompt = self.agent.generate_reflection_prompt()
            strategy_id = self.agent.current_strategy_id
            
            self.agent.update_short_term_memory("System", reflection_prompt)
            response = self.agent.think(reflection_prompt)
            self.agent.update_short_term_memory("Agent", response)
            
            # Now this will work because the method exists in ThinkingAgent
            self.agent.evaluate_prompt_effectiveness(reflection_prompt, response, strategy_id)
            
            console.print(Panel(response, title="[bold blue]Background Reflection[/bold blue]", border_style="green"))
        
        threading.Thread(target=background_reflection, daemon=True).start()
        self.root.after(10000, self.schedule_reflection)

    def run(self):
        self.root.mainloop()

def get_available_databases():
    """Get list of existing database files"""
    return [f for f in os.listdir() if f.endswith('.db')]

def choose_agent_database():
    """Pick an existing agent database or name a new one. Returns (db_path, is_new)."""
    existing_dbs = get_available_databases()
    if existing_dbs:
        console.print("\n[bold cyan]Found existing agent databases:[/bold cyan]")
        for i, db in enumerate(existing_dbs, 1):
            console.print(f"{i}. {db}")
        console.print("n. Create new agent")
        choice = Prompt.ask(
            "Choose an option",
            choices=[str(i) for i in range(1, len(existing_dbs) + 1)] + ['n'],
            default="n"
        )
        if choice != 'n':
            return existing_dbs[int(choice) - 1], False
        db_name = Prompt.ask("\nEnter name for new agent database", default="new_agent.db")
        if not db_name.endswith('.db'):
            db_name += '.db'
        return db_name, True
    console.print("\n[cyan]Creating new agent...[/cyan]")
    return "llm_memory.db", True

def choose_provider_name(saved=None):
    """Ask which engine to use, marking the ones with an API key already set."""
    names = list(PROVIDERS)
    console.print("\n[bold cyan]Which engine should this agent think with?[/bold cyan]")
    for i, name in enumerate(names, 1):
        cls = PROVIDERS[name]
        has_key = bool(os.environ.get(cls.env_var)) or (name == "gemini" and os.environ.get("GOOGLE_API_KEY"))
        note = " [green](API key found)[/green]" if has_key else ""
        console.print(f"{i}. {cls.display_name}{note}")
    default = str(names.index(saved) + 1) if saved in names else "1"
    choice = Prompt.ask("Choose an engine", choices=[str(i) for i in range(1, len(names) + 1)], default=default)
    return names[int(choice) - 1]

def get_api_key(provider_name):
    cls = PROVIDERS[provider_name]
    key = os.environ.get(cls.env_var) or (os.environ.get("GOOGLE_API_KEY") if provider_name == "gemini" else None)
    if key:
        console.print(f"[dim]Using API key from ${cls.env_var}.[/dim]")
        return key
    return getpass(f"\nEnter your {cls.display_name} API key (input hidden): ").strip()

def choose_model(label, available, preferred=None):
    """Use the preferred model if the account has it; otherwise pick from the live list."""
    if preferred and preferred in available:
        if Prompt.ask(f"Use [bold]{preferred}[/bold] as the {label}?", choices=["y", "n"], default="y") == "y":
            return preferred
    elif preferred:
        console.print(f"[yellow]{preferred} is not available on this account.[/yellow]")
    console.print(f"\n[bold cyan]Available models for the {label}:[/bold cyan]")
    for i, model_id in enumerate(available, 1):
        console.print(f"{i}. {model_id}")
    choice = Prompt.ask("Choose a model (number or exact ID)")
    if choice.isdigit() and 1 <= int(choice) <= len(available):
        return available[int(choice) - 1]
    return choice.strip()

def setup_provider(agent):
    """Pick the engine and models for an agent, reusing what it saved last time."""
    saved_provider = agent.get_setting("provider")
    saved_model = agent.get_setting("model")
    saved_light = agent.get_setting("light_model")

    use_saved = False
    if saved_provider in PROVIDERS and saved_model:
        console.print(
            f"\nThis agent last ran on [bold]{PROVIDERS[saved_provider].display_name}[/bold] "
            f"with [bold]{saved_model}[/bold] (light model: {saved_light or saved_model})."
        )
        use_saved = Prompt.ask("Keep that setup?", choices=["y", "n"], default="y") == "y"

    provider_name = saved_provider if use_saved else choose_provider_name(saved_provider)
    api_key = get_api_key(provider_name)
    provider = create_provider(provider_name, api_key=api_key)

    if use_saved:
        provider.model, provider.light_model = saved_model, saved_light or saved_model
    else:
        available = provider.list_models()  # also validates the API key
        console.print("\n[green]✓ API key validated successfully![/green]")
        cls = PROVIDERS[provider_name]
        provider.model = choose_model("main thinking model", available, cls.default_model)
        provider.light_model = choose_model(
            "light model (scores reflection prompts)", available, cls.default_light_model or provider.model
        )
        agent.set_setting("provider", provider_name)
        agent.set_setting("model", provider.model)
        agent.set_setting("light_model", provider.light_model)

    agent.provider = provider
    console.print(f"\n[green]✓ Thinking with {provider.display_name}: {provider.model}[/green]")
    return provider

def initialize_agent():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel.fit(
        "[bold blue]Welcome to LLM Life Agent[/bold blue]\n[italic]A self-reflective AI companion[/italic]",
        border_style="bold blue")
    )

    try:
        db_path, is_new = choose_agent_database()
        agent = ThinkingAgent(None, db_path=db_path)
        setup_provider(agent)

        if is_new:
            console.print("\n[bold cyan]Let's configure your new AI agent![/bold cyan]")
            configure_new_agent(agent)
        elif agent.load_active_identity():
            console.print("\n[green]✓ Loaded existing agent configuration![/green]")
        else:
            console.print("\n[yellow]No active configuration found. Let's create one![/yellow]")
            configure_new_agent(agent)

        return agent

    except Exception as e:
        console.print(f"\n[red]✗ Error: {str(e)}[/red]")
        return None

def configure_new_agent(agent):
    """Configure a new agent's identity"""
    console.print("\nWhat is the primary purpose of this agent? (e.g., 'Help me learn Spanish', 'Assist with creative writing')", style="bold")
    purpose = input().strip()
    
    console.print("\nDescribe the agent's personality (or press Enter for default):", style="cyan")
    console.print("Example: 'Patient and encouraging' or 'Analytical and direct'", style="dim")
    personality = input().strip()
    
    console.print("\nSpecify any particular goals (or press Enter for default):", style="cyan")
    console.print("Example: 'Focus on practical conversation skills' or 'Emphasize creative storytelling'", style="dim")
    goals = input().strip()
    
    console.print("\nAny constraints or specific guidelines? (or press Enter for default):", style="cyan")
    console.print("Example: 'Keep responses concise' or 'Always include examples'", style="dim")
    constraints = input().strip()
    
    # Store the configuration
    conn = sqlite3.connect(agent.db_path)
    cursor = conn.cursor()
    
    # Deactivate any existing configurations
    cursor.execute("UPDATE agent_identity SET is_active = 0 WHERE is_active = 1")
    
    # Insert new configuration
    cursor.execute("""
    INSERT INTO agent_identity (purpose, personality, goals, constraints)
    VALUES (?, ?, ?, ?)
    """, (purpose, personality, goals, constraints))
    
    conn.commit()
    conn.close()
    
    # Load the new configuration
    agent.load_active_identity()
    console.print("\n[green]✓ Agent configured successfully![/green]")

def main():
    while True:
        agent = initialize_agent()  # This creates the ThinkingAgent
        if agent:
            break
        if Prompt.ask("\nWould you like to try again?", choices=["y", "n"], default="y") != "y":
            console.print("\n[yellow]Goodbye![/yellow]")
            return

    # Create and run the GUI
    gui = SimpleInputGUI(agent)
    try:
        gui.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Reflection cycle interrupted. Thank you![/yellow]")

if __name__ == "__main__":
    main()