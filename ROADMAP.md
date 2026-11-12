# Roadmap

Known issues that need design work before they're fixed, with the options
considered so far. Recommended options are marked **(recommended)**.

## 1. Running costs

**Problem.** The background reflection loop runs every 10 seconds whether or
not anyone is using the agent. Each cycle makes 1 call to write the reflection
prompt, 1 to 6 calls to think (one per tool round), and 1 call to score the
prompt. That's about 360 cycles an hour with no spending limit and no
visibility into spend.

**Already done.** Scoring runs on each provider's cheaper "light" model, and the
default Claude setup uses Claude Opus 5.5 at `medium` effort for thinking and
Claude Haiku 5.5 for scoring.

**Options.**
1. **Slow down and pause when idle (recommended).** Make the reflection
   interval configurable (default 2–5 minutes) and pause automatically after N
   minutes with no user input. This alone cuts spend by 90% or more.
2. **Track usage and set a budget (recommended).** Every provider reports
   token counts on its responses. Record them in a `usage` table (provider,
   model, input/output tokens, estimated cost), show the running total in the
   GUI, and stop reflecting at an hourly or daily limit.
3. **Per-task model routing.** Extend the main/light split to more roles, for
   example writing reflection prompts on the light model too. Measure quality
   before moving core thinking to a smaller model.
4. **Provider cost features.** Prompt caching (the system prompt and tool list
   are identical every cycle, so they cache well) and batch APIs for
   non-urgent work like scoring.

## 2. The agent grades itself

**Problem.** The same identity writes the reflection prompt, answers it, and
scores the result, so scores probably lean optimistic and reward whatever style
the agent already likes.

**Already done.** Scoring runs on the light model, which is a separate model but
still gets the agent's identity as its system prompt.

**Options.**
1. **Separate grader (recommended).** A different system prompt with no agent
   identity, scoring against a fixed rubric and returning a score per criterion
   (depth, actionability, progress toward purpose, novelty) instead of one
   number. Optionally use a different provider entirely.
2. **Measured signals (recommended).** Combine the grader's score with things
   that can be counted: rows added, tool calls that succeeded, new tables,
   near-duplicate thoughts (a penalty).
3. **User feedback.** 👍/👎 buttons in the GUI that adjust the current
   strategy's score, weighted above the model's own rating.
4. **Exploration.** Most of the time, build on the best-scoring prompts;
   occasionally try something deliberately different, so the agent doesn't keep
   refining one style.

## 3. No safety limits on the database

**Problem.** The model can run any SQL. One bad migration (`DROP TABLE`,
a broken `ALTER`) can wipe the agent's memory, and `query_data` can change
data even though it's meant to be read-only.

**Options (all recommended, each small).**
1. **Back up before migrations.** Before any `execute_migration`, copy the
   `.db` file with SQLite's backup API and keep the last N copies. You can
   always undo, and the agent isn't limited at all.
2. **Protect the core tables.** Refuse `DROP`/`ALTER` on `agent_identity`,
   `agent_settings`, `reflection_strategies`, `schema_history` and `thoughts`.
   Check statements with `sqlite3.Connection.set_authorizer` rather than text
   matching.
3. **Read-only queries.** Run `query_data` on a read-only connection
   (`file:...?mode=ro` URI).
4. **Separate store from migrate.** `store_data` should accept only single
   `INSERT` statements, with schema changes kept on `execute_migration`.

## 4. Verify OpenAI and Gemini model defaults

OpenAI and Gemini have no hard-coded default models: at setup the agent lists
the models your account can use and you pick one, and the agent saves the
choice. Once current model IDs and prices are confirmed against
the vendors' official pricing pages, set `default_model` /
`default_light_model` on `OpenAIProvider` and `GeminiProvider` in
`providers.py`, as `AnthropicProvider` does.

## 5. Autonomous self-evaluation and development

**Design principle.** Growing Agent is responsible for directing and evaluating its own development. The user may define an agent's initial purpose and constraints, but the agent should identify what progress means in relation to that purpose, devise ways to measure it, and revise those measures as its understanding evolves. Progress measurement is part of the agent's autonomy, not a developer-administered scorecard.

**Current state.** The agent generates reflection prompts, scores their effectiveness, stores scores, and uses high- and low-scoring prompts when generating future reflections. This is a starting feedback loop, not yet a general system for measuring development. Its reflection-scoring rubric is currently fixed in code, and a score is the agent's assessment rather than independently established proof of progress.

**Planned capabilities.**
1. **Self-directed objectives.** From its configured purpose and accumulated experience, identify concrete goals, priorities, limitations, and opportunities for development.
2. **Self-generated measures.** Decide what observable evidence would indicate progress toward each objective. Define a baseline, expected change, and measurement method; revise these when experience shows they are inadequate.
3. **Hypotheses and experiments.** Propose changes to strategies, knowledge structures, or methods; state the expected benefit; test and compare outcomes.
4. **Evidence-based judgments.** Record observations and measurements separately from predictions, interpretations, and self-assigned scores. Preserve unsuccessful and inconclusive results, not just successes.
5. **Evolving evaluation.** Reflect on whether its chosen measurements genuinely correspond to its purpose, and improve the evaluation method itself when warranted.
6. **Persistent development history.** Preserve objectives, hypotheses, baselines, measurement definitions, experiments, results, lessons, and revisions so the agent can trace why it believes it has improved or regressed.

**What would count as progress?** An observed change in the agent's ability to achieve something relevant to its purpose, compared with an earlier baseline, with supporting evidence the agent can inspect and reassess. More thoughts, database entries, or high self-ratings alone do not establish improvement. The agent should also be able to recognize when evidence is insufficient.

**Relationship to existing issues.** The separate-grader and measured-signal options in section 2 can improve the reliability of reflection scores, but they do not replace the broader autonomous evaluation process described here. Database safety and sustainable running costs in sections 1 and 3 remain prerequisites for reliable long-term experimentation.

## Not planned: Cursor as an engine

Cursor is an editor, not a model provider. It has no general-purpose
inference API, and its Cloud Agents API runs Cursor's own coding agents rather
than answering model calls. The models Cursor uses are available directly from
Anthropic, OpenAI and Google, and those three are supported here.
