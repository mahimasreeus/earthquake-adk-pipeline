"""ADK agent definitions.

    root_agent (SequentialAgent "PipelineOrchestrator")
      ├─ IngestionAgent   (LlmAgent)  request -> tool params, fetch raw data
      ├─ ProcessingAgent  (LlmAgent)  clean, aggregate, quality-gate
      ├─ StorageAgent     (LlmAgent)  upload raw + processed outputs
      └─ ReportingAgent   (LlmAgent)  write summary, render + publish HTML

The SequentialAgent fixes the stage order, so the pipeline is predictable and
easy to audit. Each stage is an LLM agent that decides how to call its tools,
reads their results, retries or halts on problems, and reports back. Data
moves between stages as files; only small values (paths, counts, run_id,
KPIs) travel through session state.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.genai import types

from .config import get_settings
from .tools.common import ensure_run_context
from .tools.control import HALT_REASON, HALTED, halt_pipeline
from .tools.ingestion import fetch_earthquake_data
from .tools.processing import clean_and_transform, compute_aggregations, run_quality_checks
from .tools.reporting import generate_html_report, get_report_data, upload_report
from .tools.storage import upload_processed, upload_raw

settings = get_settings()
# Retry transient Gemini errors (429 rate limit, 5xx "high demand") with
# exponential backoff, and cap each request at 60 s so a stuck call can't
# hang the whole run.
MODEL = Gemini(
    model=settings.gemini_model,
    client_kwargs={"http_options": types.HttpOptions(timeout=60_000)},
    retry_options=types.HttpRetryOptions(
        attempts=5, initial_delay=2.0, max_delay=30.0, exp_base=2.0,
        http_status_codes=[429, 500, 502, 503, 504]),
)
# Low temperature: these agents follow procedures; they don't need creativity.
GEN_CONFIG = types.GenerateContentConfig(temperature=0.2)

COMMON_RULES = """
General rules:
- Every tool returns a dict with "status": "success" or "error". Always read it.
- If a tool returns "error", retry it once only if the error looks transient
  (network/timeout). If it still fails, call halt_pipeline with a one-sentence
  reason and stop.
- Never invent numbers, file paths, or URIs: report only what tools returned.
- Keep your final response short (2-4 lines) and factual.
"""


# --- Callbacks (deterministic guardrails around the LLM agents) -----------
# Per-run state keys. Cleared at the start of every run so a second request
# in the same `adk web` session never sees the previous run's results.
RUN_STATE_KEYS = (
    "run_id", "run_date", "run_dir", "raw_path", "raw_count", "clean_path",
    "raw_rows", "clean_rows", "kpis_path", "quality_path", "quality_status",
    "raw_uris", "processed_uris", "report_path", "local_latest_report",
    "report_uris", "query_start", "query_end", "min_magnitude", "source_url",
    "ingestion_summary", "processing_summary", "storage_summary", "report_summary",
)


def start_run(callback_context: CallbackContext) -> None:
    """Root before-callback: start a fresh run with a new run_id so every
    artifact is traceable to exactly one run."""
    state = callback_context.state
    for key in RUN_STATE_KEYS:
        if state.get(key) is not None:
            state[key] = None
    state[HALTED] = False
    state[HALT_REASON] = ""
    ensure_run_context(state)
    return None


def skip_if_halted(callback_context: CallbackContext) -> types.Content | None:
    """Sub-agent before-callback: skip this stage if an earlier stage halted
    the pipeline. Also a deterministic backstop: never publish data whose
    quality check failed, even if the LLM forgot to halt."""
    state = callback_context.state
    if state.get("quality_status") == "fail" and not state.get(HALTED):
        state[HALTED] = True
        state[HALT_REASON] = "Quality checks failed (guardrail)."
    if state.get(HALTED):
        reason = state.get(HALT_REASON) or "an earlier stage stopped the run"
        return types.Content(role="model", parts=[types.Part(
            text=f"Skipped: pipeline halted — {reason}")])
    return None


# --- Specialist agents -----------------------------------------------------
ingestion_agent = LlmAgent(
    name="IngestionAgent",
    model=MODEL,
    description="Turns the user's request into USGS query parameters and fetches raw earthquake data.",
    instruction=f"""You are the ingestion stage of an earthquake data pipeline.

1. Read the user's request and decide the query parameters for
   fetch_earthquake_data:
   - days: the look-back window. "last 2 weeks" = 14, "this past week" = 7.
     If the user does not say, use {settings.days}.
   - min_magnitude: "magnitude 4+" or "M4 and above" = 4.0. If the user does
     not say, use {settings.min_magnitude}.
   - start_date / end_date (YYYY-MM-DD): only if the user gave explicit
     calendar dates; otherwise leave them out.
   - Requests for more than 366 days are not supported: use 366 and say so.
2. Call fetch_earthquake_data exactly once with those parameters (plus at most
   one retry on a transient error).
3. Check the result: if record_count is 0, call halt_pipeline explaining that
   there is nothing to process. If there is a warning, mention it.
4. Final response: the window, minimum magnitude, record count, and whether
   the fallback feed was used.
{COMMON_RULES}""",
    tools=[fetch_earthquake_data, halt_pipeline],
    output_key="ingestion_summary",
    generate_content_config=GEN_CONFIG,
    before_agent_callback=skip_if_halted,
)

processing_agent = LlmAgent(
    name="ProcessingAgent",
    model=MODEL,
    description="Cleans and transforms raw data, computes aggregates, and gates on data quality.",
    instruction=f"""You are the processing stage. The raw data was already fetched;
its location is in session state, so your tools take no data arguments.

1. Call clean_and_transform. Check clean_rows > 0 and look at how many rows
   each step removed.
2. Call compute_aggregations.
3. Call run_quality_checks and act on quality_status:
   - "pass": continue.
   - "warn": continue, but list each issue in your response.
   - "fail": call halt_pipeline with the issues as the reason. Do not continue.
4. Final response: raw rows -> clean rows, rows removed per step, quality
   status (and issues), and 2-3 headline KPIs (total events, strongest event).
{COMMON_RULES}""",
    tools=[clean_and_transform, compute_aggregations, run_quality_checks, halt_pipeline],
    output_key="processing_summary",
    generate_content_config=GEN_CONFIG,
    before_agent_callback=skip_if_halted,
)

storage_agent = LlmAgent(
    name="StorageAgent",
    model=MODEL,
    description="Uploads raw and processed data to the configured storage backend.",
    instruction=f"""You are the storage stage. Persist this run's data to the
configured backend (local folder, Google Cloud Storage, or S3).

1. Call upload_raw and upload_processed. They are independent, so request
   both in the same turn.
2. Final response: the backend name, how many files were written, and the
   folder prefix they were written under (copy it from the returned URIs).
   Do not list every URI.
{COMMON_RULES}""",
    tools=[upload_raw, upload_processed, halt_pipeline],
    output_key="storage_summary",
    generate_content_config=GEN_CONFIG,
    before_agent_callback=skip_if_halted,
)

reporting_agent = LlmAgent(
    name="ReportingAgent",
    model=MODEL,
    description="Writes a plain-English summary and publishes the HTML report.",
    instruction=f"""You are the reporting stage. You turn this run's results into a
short, accurate written summary and publish the HTML report.

1. Call get_report_data to get the KPIs, top regions, magnitude band x depth
   counts, and the quality result.
2. Write a 3-5 sentence plain-English summary for a general audience, using
   only numbers from get_report_data. Cover: overall volume and the query
   window, the strongest event (magnitude, place, date), where activity was
   concentrated (top regions), the depth pattern (share of shallow events),
   and anything notable such as tsunami flags or PAGER alerts. No markdown,
   no bullet points.
   Accuracy rule: only say a specific event had a tsunami flag or PAGER alert
   if that event's own fields show it (strongest_event.tsunami_flag,
   strongest_event.pager_alert, or tsunami_flagged_events). Otherwise state
   counts without attributing them to an event.
3. Call generate_html_report with summary_text set to that summary.
4. Call upload_report.
5. Final response: the summary you wrote, then the report URIs from
   upload_report on separate lines.
{COMMON_RULES}""",
    tools=[get_report_data, generate_html_report, upload_report, halt_pipeline],
    output_key="report_summary",
    generate_content_config=GEN_CONFIG,
    before_agent_callback=skip_if_halted,
)

# --- Orchestrator ----------------------------------------------------------
root_agent = SequentialAgent(
    name="PipelineOrchestrator",
    description="Runs ingestion -> processing -> storage -> reporting for USGS earthquake data.",
    sub_agents=[ingestion_agent, processing_agent, storage_agent, reporting_agent],
    before_agent_callback=start_run,
)
