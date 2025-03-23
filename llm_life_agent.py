import openai
from openai import OpenAI
import sqlite3
from datetime import datetime
import os
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import tkinter as tk
import threading

console = Console()

class ThinkingAgent:
    def __init__(self, client, db_path="llm_memory.db"):
        self.client = client
        self.db_path = db_path
        self.short_term_memory = []  # STM buffer
        self.system_message = {"role": "system", "content": "You are a self-reflective AI agent with the ability to learn and adapt."}
        self.last_tool_results = []
        self.user_input_queue = None  # Add this to store user input
        self.initialize_database()
        # Add table for storing reflection strategies
        self.initialize_reflection_table()
        self.initialize_thought_tables()
        self.initialize_agent_identity()
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
        client = OpenAI()
        try:
            response = client.responses.create(
                model="gpt-4o",
                input=query,
                tools=[
                    {"type": "web_search_preview"}
                ]
            )
            return {
                "success": True,
                "results": response.output_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def make_openai_call(self, messages, tools=None):
        """Helper method to standardize OpenAI API calls"""
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
            return response
        except Exception as e:
            console.print(f"[red]Error in OpenAI API call: {str(e)}[/red]")
            return None

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

        messages = [
            self.system_message,
            {"role": "user", "content": f"""
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
                
                IMPORTANT: Your response MUST include at least one insight or observation that should be stored in the database.
                Include specific questions that will help drive progress toward your primary purpose.
            """}
        ]

        response = self.make_openai_call(messages)
        if not response:
            return "Failed to generate reflection prompt"
        
        generated_prompt = response.choices[0].message.content

        # Store the generated prompt
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO reflection_strategies (prompt_template, effectiveness_score)
        VALUES (?, 0.0)
        """, (generated_prompt,))
        conn.commit()
        conn.close()

        return generated_prompt

    def think(self, prompt):
        # Add user input to prompt if available
        if self.user_input_queue:
            prompt = f"{prompt}\n\nUser Input: {self.user_input_queue}"
            self.user_input_queue = None  # Clear the queue after using it
            
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_migration",
                    "description": "Execute a SQL migration to modify the database schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {"type": "string", "description": "SQL migration command"}
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "store_data",
                    "description": "Store any type of data in the database. Use this to record any structured data in your tables.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string", 
                                "description": "SQL INSERT command for any table. Examples:\n" +
                                             "- INSERT INTO opportunities (name, potential_value) VALUES ('New Market', 50000)\n" +
                                             "- INSERT INTO metrics (category, value) VALUES ('ROI', 12.5)\n" +
                                             "- INSERT INTO thoughts (content) VALUES ('Strategic insight about...')"
                            }
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_schema",
                    "description": "Get the current database schema",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_data",
                    "description": "Query any data from the database tables using SELECT statements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string", 
                                "description": "SQL SELECT query. Examples:\n" +
                                             "- SELECT * FROM thoughts ORDER BY timestamp DESC LIMIT 5\n" +
                                             "- SELECT category, COUNT(*) FROM metrics GROUP BY category\n" +
                                             "- SELECT * FROM opportunities WHERE potential_value > 1000"
                            }
                        },
                        "required": ["sql"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet for current information about markets, opportunities, and trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up information online"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        schema_text = self.get_latest_schema_text()
        short_term_context = "\n".join([f"{entry['speaker']}: {entry['message']}" for entry in self.short_term_memory])
        
        # Include any previous tool results in the context
        tool_results_context = ""
        if self.last_tool_results:
            tool_results_context = "\nPrevious tool execution results:\n" + "\n".join(self.last_tool_results)
        
        messages = [
            {"role": "system", "content": "You are a self-reflective AI agent. Your primary purpose is to become as interesting as possible."},
            {"role": "user", "content": f"{prompt}\n\nShort-Term Memory Context:\n{short_term_context}{tool_results_context}"}
        ]

        # Reset tool results for this iteration
        self.last_tool_results = []

        response = self.make_openai_call(messages, tools=tools)
        if not response:
            return "Failed to process thought"
        
        message = response.choices[0].message
        formatted_response = ""
        stored_something = False  # Track if we've stored anything

        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except Exception:
                    function_args = {}

                if function_name == "execute_migration":
                    result = self.execute_sql(function_args.get("sql", ""))
                elif function_name == "store_data":
                    result = self.execute_sql(function_args.get("sql", ""))
                    if result.get("success", False):
                        stored_something = True
                elif function_name == "get_schema":
                    result = self.get_schema()
                elif function_name == "query_data":
                    result = self.query_database(function_args.get("sql", ""))
                elif function_name == "web_search":
                    result = self.perform_web_search(function_args.get("query", ""))
                else:
                    result = {"error": "Unknown function"}

                # Store the result for next iteration
                result_str = f"Tool {function_name}: {json.dumps(result, indent=2)}"
                self.last_tool_results.append(result_str)
                formatted_response += f"\nTool Execution Result ({function_name}):\n{json.dumps(result, indent=2)}"

                messages.append(message.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result)
                })

        # If nothing was stored, force a thought storage
        if not stored_something and message.content:
            try:
                # Store the response as a thought
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO thoughts (content, category, confidence)
                    VALUES (?, 'reflection', 0.5)
                """, (message.content[:500],))  # Limit content length
                conn.commit()
                conn.close()
                formatted_response += "\n[System: Stored response as thought]"
            except Exception as e:
                formatted_response += f"\n[System: Failed to store thought: {str(e)}]"

        if message.content:
            formatted_response += f"\nAssistant Response:\n{message.content}"

        return formatted_response

    def evaluate_prompt_effectiveness(self, prompt, response):
        messages = [
            self.system_message,
            {"role": "user", "content": f"""
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
            """}
        ]

        api_response = self.make_openai_call(messages)
        if not api_response:
            return
        
        try:
            score = float(api_response.choices[0].message.content.strip())
            
            # Update the effectiveness score in the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE reflection_strategies 
                SET effectiveness_score = (effectiveness_score * times_used + ?) / (times_used + 1),
                    times_used = times_used + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE prompt_template = ?
            """, (score, prompt))
            conn.commit()
            conn.close()
        except ValueError:
            pass

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
            
            self.agent.update_short_term_memory("System", reflection_prompt)
            response = self.agent.think(reflection_prompt)
            self.agent.update_short_term_memory("Agent", response)
            
            # Now this will work because the method exists in ThinkingAgent
            self.agent.evaluate_prompt_effectiveness(reflection_prompt, response)
            
            console.print(Panel(response, title="[bold blue]Background Reflection[/bold blue]", border_style="green"))
        
        threading.Thread(target=background_reflection, daemon=True).start()
        self.root.after(10000, self.schedule_reflection)

    def run(self):
        self.root.mainloop()

def get_available_databases():
    """Get list of existing database files"""
    return [f for f in os.listdir() if f.endswith('.db')]

def initialize_agent():
    os.system('cls' if os.name == 'nt' else 'clear')
    console.print(Panel.fit(
        "[bold blue]Welcome to LLM Life Agent[/bold blue]\n[italic]A self-reflective AI companion[/italic]",
        border_style="bold blue")
    )
    
    console.print("\nPlease enter your OpenAI API key:", style="bold")
    api_key = input().strip()
    
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        console.print("\n[green]✓ API key validated successfully![/green]")
        
        # Check for existing databases
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
            
            if choice == 'n':
                # Create new agent with new database
                db_name = Prompt.ask("\nEnter name for new agent database", default="new_agent.db")
                if not db_name.endswith('.db'):
                    db_name += '.db'
                agent = ThinkingAgent(client, db_path=db_name)
                
                # Configure new agent
                console.print("\n[bold cyan]Let's configure your new AI agent![/bold cyan]")
                configure_new_agent(agent)
            else:
                # Load existing agent
                db_path = existing_dbs[int(choice) - 1]
                agent = ThinkingAgent(client, db_path=db_path)
                
                if agent.load_active_identity():
                    console.print("\n[green]✓ Loaded existing agent configuration![/green]")
                else:
                    console.print("\n[yellow]No active configuration found. Let's create one![/yellow]")
                    configure_new_agent(agent)
        else:
            # No existing databases, create new
            console.print("\n[cyan]Creating new agent...[/cyan]")
            agent = ThinkingAgent(client)
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