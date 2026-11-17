"""CLI entry point: run the agentic earthquake pipeline end to end.

    python main.py                              # last 30 days, M2.5+
    python main.py --days 14 --min-magnitude 4
    python main.py --backend gcs
    python main.py --prompt "Show me M5+ quakes from the past week"

Creates an ADK Runner with an in-memory session, sends one request to the
root agent, streams each agent/tool event, and prints where outputs went.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings

APP_NAME = "earthquake_pipeline"
USER_ID = "cli_user"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic USGS earthquake pipeline (Google ADK).")
    parser.add_argument("--days", type=int, default=int(os.getenv("PIPELINE_DAYS", 30)),
                        help="look-back window in days (default 30)")
    parser.add_argument("--min-magnitude", type=float,
                        default=float(os.getenv("PIPELINE_MIN_MAGNITUDE", 2.5)),
                        help="minimum magnitude (default 2.5)")
    parser.add_argument("--backend", choices=["local", "gcs", "s3"],
                        help="storage backend (overrides STORAGE_BACKEND)")
    parser.add_argument("--prompt", help="free-text request instead of --days/--min-magnitude")
    parser.add_argument("--verbose", action="store_true", help="show tool INFO logs")
    return parser.parse_args()


def describe_event(event) -> list[str]:
    """Turn one ADK event into short, readable trace lines."""
    lines = []
    for call in event.get_function_calls() or []:
        args = json.dumps(call.args or {}, default=str)
        lines.append(f"  [{event.author}] -> tool {call.name}({args})")
    for resp in event.get_function_responses() or []:
        result = resp.response or {}
        status = result.get("status", "?")
        detail = result.get("error") or result.get("warning") or ""
        lines.append(f"  [{event.author}] <- {resp.name}: {status}"
                     + (f" ({detail})" if detail else ""))
    if event.is_final_response() and event.content and event.content.parts:
        text = "".join(p.text or "" for p in event.content.parts).strip()
        if text:
            lines.append(f"\n[{event.author}] {text}\n")
    return lines


async def run_pipeline(request: str) -> int:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from earthquake_pipeline.agent import root_agent

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    print(f"Request: {request}\n")
    message = types.Content(role="user", parts=[types.Part(text=request)])
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session.id,
                                            new_message=message):
            for line in describe_event(event):
                print(line, flush=True)
    except Exception as exc:  # e.g. Gemini still unavailable after retries
        print(f"\nERROR: the run stopped early: {type(exc).__name__}: {str(exc)[:300]}",
              file=sys.stderr)
        print("Outputs from completed stages are still listed below. "
              "If this was a 429/503 from Gemini, wait a minute and re-run.", file=sys.stderr)

    state = (await session_service.get_session(app_name=APP_NAME, user_id=USER_ID,
                                               session_id=session.id)).state
    return print_summary(state)


def print_summary(state: dict) -> int:
    print("=" * 72)
    print(f"Run ID:          {state.get('run_id')}")
    print(f"Quality status:  {state.get('quality_status')}")
    if state.get("pipeline_halted"):
        print(f"PIPELINE HALTED: {state.get('halt_reason')}")
    uris = [u for key in ("raw_uris", "processed_uris", "report_uris") for u in (state.get(key) or [])]
    if uris:
        print("Objects written:")
        for uri in uris:
            print(f"  {uri}")
    if state.get("local_latest_report"):
        print(f"Open the report: {state['local_latest_report']}")
    print("=" * 72)
    ok = bool(state.get("report_uris")) and not state.get("pipeline_halted")
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    if args.backend:
        os.environ["STORAGE_BACKEND"] = args.backend
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)  # quiet SDK experimental notices

    from earthquake_pipeline.config import ConfigError, get_settings, validate_llm_credentials

    try:
        validate_llm_credentials()
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"Model: {settings.gemini_model} | storage backend: {settings.storage_backend}")
    request = args.prompt or (
        f"Run the earthquake pipeline for the last {args.days} days "
        f"with minimum magnitude {args.min_magnitude}.")
    return asyncio.run(run_pipeline(request))


if __name__ == "__main__":
    sys.exit(main())
