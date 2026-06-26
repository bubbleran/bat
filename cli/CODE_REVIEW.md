# bat-cli — Code Review

Review of the whole CLI (`cli/`), cross-checked against the `bat-adk` package
(`adk/`). Findings are grouped by subsystem. Each carries a severity and a
status:

- ✅ **fixed** in this branch (see the *Fixes applied* section at the bottom)
- ⬜ **open** — not yet addressed

Severity: **critical** (broken core promise) · **major** · **minor** · **nit**.

> Test/lint baseline at review time: the entire CLI suite was failing
> (36/43 errors, all `CliRunner.isolated_filesystem` — see Scaffold/M1), there
> is no CI running ruff/pytest, and ruff reports 110 pre-existing `E501`
> line-length violations.

---

## 1. Scaffold — `init agent` / `add client` / `set env`

Files: `src/cli.py`, `src/create/agent.py`, `src/add/client.py`,
`src/set/env.py`, `src/create/templates/agent/*`.

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| C1 | critical | ✅ fixed | Template `config.yaml` `telemetry.output:` parses as `None`; adk's `TelemetrySettings.output: List[...] = Field(default=[])` rejects it → **every freshly scaffolded agent crashes at startup** (`AgentConfig.load()` raises). |
| C2 | critical | ✅ fixed* | Client class names diverge between the written client file and the regenerated `llm_clients/__init__.py` (`RNG` → file `RNGClient`, init imports `RngClient`) → `ImportError` at agent startup; `add client` can corrupt a working agent. (*Fixed as a side effect of M2.) |
| M1 | major | ✅ fixed | Whole test suite fails: `CliRunner` (typer ≥0.13) no longer has `isolated_filesystem`. 36/43 tests error. |
| M2 | major | ✅ fixed | `llm_clients/__init__.py` regenerated unconditionally (ignores `--force`) and an import is generated for **every** `.py` in the dir (e.g. a `utils.py` helper → `ImportError`). |
| M3 | major | ⬜ open | `--force` help says it "delete existing files" (it never deletes); it silently overwrites the user's edited `.env` (wiping the API key) and `config.yaml`/`graph.py` with no backup or warning. |
| M4 | major | ✅ fixed | Agent `name` used as a raw path: `../foo`, absolute paths and nested paths escape `--output-dir`; `""` scaffolds into cwd. |
| M5 | major | ✅ fixed | Uncaught exceptions → tracebacks: target-exists-as-file (`NotADirectoryError`) and malformed `config.yaml` (`yaml.YAMLError`). |
| M6 | major | ✅ fixed | `set env` destroys all comments/structure in `config.yaml` (full `safe_load`→`safe_dump` round-trip) and silently discards non-mapping content. |
| M7 | major | ✅ fixed | Scaffold has no `.gitignore` (so `git add .` commits the API key) and `.dockerignore` doesn't exclude `.env` (key baked into builder layers). |
| M8 | major | ✅ fixed | (adk) Telemetry enabled in `config.yaml` but the `bat-adk[telemetry]` extra absent → `RuntimeError` propagates as a raw traceback through `AgentApplication.__init__`. Now logs a clear "no telemetry installed" error and continues with telemetry disabled. |
| m1 | minor | ⬜ open | `set/env.py:_upsert_env_var` passes the value as a regex replacement string → `--repo 'a\1b'` raises `re.error`; backslashes corrupt the value. |
| m2 | minor | ⬜ open | Raw, unescaped substitution of user `--model`/`name` values into YAML/JSON/Makefile (e.g. `--model "foo: bar"` → invalid YAML). |
| m3 | minor | ⬜ open | "Files written" count double-counts `src/llm_clients/__init__.py`. |
| m4 | minor | ⬜ open | `--model-provider` accepts any string; unknown providers crash at runtime (adk `ModelProvider` is a closed `Literal`) and get a misleading "needs no API key" comment. |
| m5 | minor | ⬜ open | Template `Makefile` uses `DOCKER_REGISTRY`/`REPO`, ignoring the `BAT_DOCKER_*` keys `set env` writes — two build paths diverge. |
| m6 | minor | ⬜ open | `agent.spec` has stale `br_rapp_sdk` hidden imports (leftover from another SDK). |
| m7 | minor | ⬜ open | Generated `graph.py`/`llm_client.py` ship with unused imports (`ReActLoop`, `HumanMessage`) — lint errors out of the box. |
| m8 | minor | ⬜ open | Generated `.env` still says "Copy this file to .env" (it *is* `.env`). |
| m9 | minor | ⬜ open | `.python-version` (3.13) conflicts with Dockerfile base `python3.12`. |
| n1 | nit | ⬜ open | `cli.py:add client`: `_parse_clients_option(clients) or []` — the `or []` is unreachable; `add/client.py` largely duplicates the `src/llm_clients` check. |
| n2 | nit | ⬜ open | `bat set env` mostly edits `config.yaml`, not env (stale name); `--model_provider` underscore alias is unconventional. |

---

## 2. Build / Push — `build` / `push`

Files: `src/build/build.py`, `src/push/push.py`, `src/image_defaults.py`.

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| BP-M1 | major | ⬜ open | `.env` lookup uses CWD (`image_defaults.py:19`), ignoring `--context`. Running `bat build --context ./agent` from a parent dir misses the agent's `.env` and silently falls to placeholder defaults. `resolve_registry(context_dir, …)` even takes `context_dir` and never uses it. |
| BP-M2 | major | ⬜ open | Placeholder defaults `default_registry` / `default-repository/<name>` are used **silently**; with no `.` in the registry Docker treats it as docker.io → confusing `denied` errors or an unintended public push. Should abort/warn. |
| BP-M3 | major | ⬜ open | `--build-arg VERSION=…` is passed but the template `Dockerfile` has no `ARG VERSION` → no-op + `UnusedBuildArgs` warning, despite docs promising the container reports its build version. |
| BP-M4 | major | ✅ fixed | Secrets: template `.dockerignore` didn't exclude `.env`, and `Dockerfile` does `COPY . .` → API keys baked into builder-stage layers. (Fixed together with Scaffold/M7.) |
| BP-m1 | minor | ⬜ open | Docs (README/`docs/bat-cli.md`) say port/model/provider go to `.env`; code writes `config.yaml` (stale docs for `set env`, the command that feeds build/push). |
| BP-m2 | minor | ⬜ open | Inline comments not stripped from `.env` values (`KEY=host # prod` → literal `host # prod`). |
| BP-m3 | minor | ⬜ open | Only `FileNotFoundError` caught when spawning docker; `PermissionError` etc. traceback. |
| BP-m4 | minor | ⬜ open | No `--dry-run`/preview despite 4-level value resolution; you can't see the ref `push` will use before pushing. |
| BP-m5 | minor | ⬜ open | Hardcoded `docker`; no podman/engine override, no `--platform` passthrough. |
| BP-m6 | minor | ⬜ open | `build`/`push`/`eval`/`set` install as **top-level** packages; `build` shadows the PyPA `build` tool in the venv install path. |
| BP-m7 | minor | ⬜ open | `build.py`/`push.py` are near-verbatim duplicates (~40 lines: context validation, ref resolution, subprocess try/except). |
| BP-nits | nit | ⬜ open | `push.py` passes meaningless `cwd` to `docker push`; README command tree omits `version`; `make build` vs `bat build` default tags differ; "Context directory not found" message wrong when path is a file. |

**Verified non-issues:** no `shell=True` (args passed as lists → no injection); Dockerfile presence checked before build; exit codes propagate; flag/env naming consistent (`--docker-registry`↔`BAT_DOCKER_REGISTRY`); no credentials handled/logged.

---

## 3. Eval — metrics / LLM judge / plotter

Files: `src/eval/engine/evaluator.py`, `src/eval/engine/metrics/*`,
`src/eval/engine/plotter.py`. *(Findings empirically verified by the reviewer.)*

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| EV-C1 | critical | ✅ fixed | `bat eval plot` crashes with `TypeError` on any null qualitative score (the default `k=1` path): `_average_episodes` returns the raw episode whose `qualitative.*` can be `None`, `.get(..., 0)` doesn't coalesce it, `axes[2].bar(...)` raises → no charts, leaked figures. The `None`-coalescing exists in `_plot_comparison` but is missing here. |
| EV-M1 | major | ✅ fixed | Judge failures are swallowed: `_call_llm_judge` returns `{"score": None, "reasoning": "Error: …"}` with **no log**, and downstream only records when `score is not None` → a misconfigured judge produces a "successful" eval with all-`None` scores and zero diagnostics. (Interacts with EV-C1.) |
| EV-M2 | major | ⬜ open | Judge LLM runs at provider-default temperature (no temp/seed/max_tokens) → nondeterministic scoring; no per-call timeout/token cap. |
| EV-M3 | major | ⬜ open | Evaluated agent output is interpolated raw into judge prompts → prompt injection (an agent can steer its own score). Only the hallucination prompt partially mitigates. |
| EV-m* | minor | ⬜ open | Scores never clamped to [0,1]; greedy list-subset matching gives order-dependent false negatives (`evaluator.py:_is_subset`); hardcoded `gpt-4.1-mini` judge fallback contradicts docstring; module-level judge-client cache (stale env + race); no backoff/brittle ``` fence stripping; no cost controls (up to ~32 concurrent LLM calls); two conflicting "success" definitions (metrics.json `pass_rate` vs plotter majority vote); `by_model` aggregation is dead code; per-task averages coalesce missing→0; figures accumulate before save (leak on exception); `tool_call_appropriateness` computed/persisted but never plotted. |
| EV-cov | — | ⬜ open | Test coverage: only `adapter` is tested. No coverage for `EpisodeEvaluator` verdict logic, `metrics.py`, `_call_llm_judge` (fence/retry/malformed JSON), `qualitative_helpers`, or the plotter `None` path — which is exactly why EV-C1 went undetected. |

**Verified OK:** evaluator/metrics field usage matches `contracts.py`; plotter sets `Agg` before importing pyplot; division-by-zero guards present; no mutable default args.

---

## 4. Eval — commands / config / runner / span integration

Files: `src/eval/commands.py`, `eval_config.py`, `contracts.py`, `adapter.py`,
`orchestrator.py`, `bench_runner.py`, cross-checked against adk telemetry
(`adk/.../telemetry/file_exporter.py`, `attributes.py`, `agent/_executor.py`).

> **Theme:** the pipeline **fails open**. When trace/conversation correlation
> or usage extraction breaks, episodes get zero tokens / no tool-calls with
> **no error** (`adapter.py:206-213` retries 20×100ms then returns empty), so
> these bugs surface as quietly-wrong metrics, never as failures. The static
> attribute-key / JSON-shape agreement between writer and reader is otherwise
> sound (keys, span-kind values, ns timestamps all line up).

| ID | Sev | Status | Issue |
|----|-----|--------|-------|
| S4-C1 | critical | ✅ fixed | Agent subprocess not reliably killed. `_start_agent_process` does `Popen(["uv","run","."])` with no `start_new_session`/process group; `_stop_agent_process` only `terminate()`/`kill()`s the **`uv` parent**, so the child python server (bound to the port) can be orphaned on timeout/Ctrl-C → port leak into the next run. `commands.py:341,352-361`. *(verified)* |
| S4-C2 | critical | ✅ fixed | Judge silently inherits the **model-under-test's `BASE_URL`**. `runner_env = server_env.copy()` carries the model's `BASE_URL`; when `judge.base_url` is unset only `JUDGE_BASE_URL` is popped, and the judge does `getenv("JUDGE_BASE_URL", getenv("BASE_URL"))` → judge calls the model's endpoint. `commands.py:553,562-565` + `llm_evaluators.py:52`. Masked by the scaffold default (judge has a base_url). *(verified)* |
| S4-C3 | critical | ✅ fixed | (adk, **uncommitted**) `_executor.py` switched `context=parent_ctx` → `links=parent_links`, so a sub-agent runs in its **own new trace_id**. The adapter groups an episode by trace_ids whose spans carry the parent's `conversation_id`, but the default `build_agent_message` (`call_agent_node.py:88-91`) sends **no `context_id`**, so the sub-agent gets a different conversation id and its trace is excluded → **multi-agent token under-counting**. The adapter comment (`adapter.py:96-100`) and the synthetic test (`test_eval_spans.py:120`) still assert the old shared-trace_id model, masking it. Single-agent eval is unaffected. *(verified)* |
| S4-M1 | major | ✅ fixed | Multi-model sweep aborts entirely on one model's failure: the `for model` loop has `try/finally` but **no `except`**, so a `_wait_for_agent_port` timeout / orchestrator error propagates out, later models never run, and no partial summary is emitted. `commands.py:491,612`. *(verified)* |
| S4-M2 | major | ⬜ open | Token **double-counting**: the usage gate counts any span with `prompt`/`completion` token attrs, not strictly `span_kind=="LLM"` (`adapter.py:129-133`). If OpenInference also stamps token counts on a parent CHAIN/AGENT span, both are summed → inflated tokens. *(unverified — needs a real span dump)* |
| S4-M3 | major | ⬜ open | `inference_time` is a **sum of span durations** over all counted spans (`adapter.py:141-145`); with parallel/nested LLM calls (or S4-M2's parent+child) durations overlap and over-sum, possibly exceeding wall-clock. *(by-design sum, undocumented)* |
| S4-M4 | major | ⬜ open | Tool-call args read from `input.value` (`adapter.py:79-86`). For LangGraph tool runs OpenInference often serializes a raw string / `{"input":...}` wrapper, not a structured args dict → `args` stays `{}` → any `args_subset` expectation silently never matches. *(unverified — tool-schema dependent)* |
| S4-M5 | major | ✅ fixed | Judge/model **provider unvalidated** against adk's closed `ModelProvider` Literal (`anthropic/deepseek/nvidia/ollama/openai`). `eval_config.py` accepts any string (and `commands.py:29-35` advertises `azure/cohere/mistral/groq` API-key envs) → a valid-looking config fails late as a per-episode judge ValidationError, not an upfront config error. |
| S4-M6 | major | ✅ fixed (isolation) | No per-task isolation/concurrency in `bench_runner`: it's a sequential `for task: for i in range(k)` with no try/except around the attempt body, so one task raising (in `evaluator.evaluate`, `task.expected`, or persist) aborts the whole run. Concurrency exists only in the qualitative phase (`orchestrator.py:45`), contrary to the "semaphore/concurrent" framing. |
| S4-m* | minor | ⬜ open | `_find_agent_python` result computed/validated then never used (agent launched via `uv run .`); dead re-validation of qualitative-without-judge (`commands.py:554-558`); `k` parsed with raw `int()` → ugly `ValueError` on `k: abc` while timeouts use a graceful parser; `_agent_url_from_config` builds `host:port:port` if `endpoint.url` already has a port, and `AttributeError` on non-string url; `_patch_agent_config` shallow-clobbers the agent's `telemetry` block (intended for isolation, undocumented); episode correlation depends on the A2A server preserving `context_id` verbatim, else silent zero-usage; `total_tokens` reconstruction (`in+out`) can drift from provider totals (cache/reasoning). |
| S4-nit | nit | ⬜ open | `eval show` omits `base_url`/`env`/judge `prompts`/timeouts (hides S4-C2 when debugging); `EpisodeTrace.usage` is an untyped `dict[str,Any]` so a key typo on either side fails silently to 0; adapter `gen_ai.tool.name` fallback is dead (never emitted on tool spans). |

**Verified OK:** writer↔reader attribute keys, span-kind values and ns timestamps all match; `contracts.py` fields are all present/consistent with what `adapter`/`bench_runner`/`orchestrator` assign; `QualitativeScores` `None`-skipping in averages is correct; the `__try{i}` suffix lives only in `conversation_id`/filenames and does **not** leak into `task_id`/summary.

**Test coverage gaps:** no teardown-on-exception / multi-model / `KeyboardInterrupt` test for `eval run`; no test where model sets `base_url` and judge omits it (S4-C2); `_resolve_env_value` braced/missing-ref forms untested; malformed `eval.yaml` paths (non-dict, `models` not a list, bad `k`) untested; `_patch/_restore_agent_config` round-trip untested; the multi-agent test asserts the now-stale shared-trace_id model (S4-C3).

---

## Fixes applied in this branch

All changes verified: **CLI suite 51/51 pass**, **adk suite 43/43 pass**.

| ID | Change |
|----|--------|
| C1 | `templates/agent/config.yaml`: `output:` → `output: []` (telemetry off by default, valid for adk). |
| M1 | All 37 `runner.isolated_filesystem()` call sites rewritten to pytest-native `tmp_path` + `monkeypatch.chdir(tmp_path)` (the idiomatic fixture pattern for current Typer, which dropped click's `isolated_filesystem`). Files re-run through `ruff format`. |
| M2 + C2 | `create/agent.py:_client_specs_from_dir` now reads the real class name from each `*_client.py` (`^class (\w+)\(`) instead of re-deriving from the filename, and only scans `*_client.py` files that define a class. Generated `__init__.py` gets an "auto-generated" header. |
| M4 | `cli.py:_validate_agent_name` rejects empty names, path separators, `..` and absolute paths. |
| M5 | `create/agent.py` raises a clean `FileExistsError` when the target is a file; `set/env.py` no longer parses YAML to edit it (no `YAMLError` path). |
| M6 | `set/env.py` edits `config.yaml` line-by-line (`_set_yaml_value`), preserving comments, blank lines and key order; inserts missing keys/sections. |
| M7 | New `templates/agent/.gitignore` (ignores `.env`, keeps `.env.template`); `.dockerignore` now excludes `.env`, `.git/`, `eval/output/`, `spans.jsonl`. |
| M8 | (adk) `telemetry/setup.py` logs a clear "no telemetry is installed" error and returns `False` (agent keeps running) instead of raising; also fixes a latent `config=None` `AttributeError`. Test updated to assert graceful degradation. |
| EV-C1 | `plotter.py:_qual_score` coalesces a missing/`None` qualitative score to `0` so the `k=1` per-task chart no longer feeds `None` into `axes.bar(...)`. Regression test `test_eval_plot_survives_null_qualitative_scores`. |
| EV-M1 | `llm_evaluators.py`: `_call_llm_judge` now logs each failed attempt + a final error; new `_apply_judge_result` helper always records the judge's reasoning (including the error text) into `judge_reasoning` and logs when a judge returns no/invalid score. Regression test `test_apply_judge_result_records_reasoning_when_score_is_none`. |
| S4-C1 | `commands.py`: agent launched with `start_new_session=True`; `_stop_agent_process` signals the whole process group (`os.killpg`, SIGTERM→SIGKILL) with a single-process fallback, so the `uv` child server is no longer orphaned. Tests `test_stop_agent_process_signals_whole_group` + `start_new_session` assertion in the run test. |
| S4-C2 | `llm_evaluators.py:_get_judge_client`: `base_url` now comes **only** from `JUDGE_BASE_URL` (dropped the `BASE_URL` fallback), so the judge never inherits the model-under-test's endpoint. Test `test_judge_client_does_not_inherit_model_base_url`. |
| S4-M5 | `eval_config.py`: model & judge `provider` validated against adk's `ModelProvider` Literal at config-load time (`_validate_provider`), turning a late per-episode `ValidationError` into a clear upfront error. Tests for unsupported model & judge providers. |
| S4-M6 | `bench_runner.py`: the per-attempt body is wrapped so a failing task is recorded as a `final_status="error"` episode (verdict failed, error in `aux`) and the run continues. Test `test_bench_runner_isolates_failing_task`. |
| S4-M1 | `commands.py`: the per-model loop body now catches exceptions, records the failed model, and continues; teardown still runs in `finally`. Final report shows `completed/total` with the failed models, and exits non-zero only if **all** models failed. Test `test_eval_run_continues_after_one_model_fails`. |
| S4-C3 | (adk) Resolved by **trace continuation** (option B): `_executor.py` reverted from `links=parent_links` back to `context=parent_ctx`, so a sub-agent continues the caller's trace (shared `trace_id`) and its tokens aggregate correctly. Chosen over propagating `context_id` because the executor reuses `context_id` as the LangGraph `thread_id` — sharing it would couple a checkpointing sub-agent's state across the caller's repeated calls. The adapter also gained a fail-open warning when an episode aggregates 0 spans. (`links_from_context` remains defined in `telemetry/setup.py` but is now unused.) |

### Notable still-open, high-value items
- **S4-M2/M4** — token double-counting on non-LLM spans; tool-call args not a structured dict (both need a real span dump to confirm).
- **EV-M2/M3** — judge runs at default temperature (nondeterministic); agent output injected raw into judge prompts.
- **BP-M1/M2** — `.env` ignores `--context`; silent placeholder image refs.
- **EV-M2/M3** — judge runs at default temperature (nondeterministic); agent output injected raw into judge prompts.
- **M3** — `--force` overwrites `.env`/`config.yaml` with no warning; help text is wrong.

> Note: several Section-4 findings (S4-C3, S4-M2, S4-M4) hinge on a real span
> dump / the in-flight telemetry refactor; verify against actual eval output
> before fixing.
