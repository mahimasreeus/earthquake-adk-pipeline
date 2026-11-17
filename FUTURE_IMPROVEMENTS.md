Repository: https://github.com/mahimasreeus/earthquake-adk-pipeline

# Future Improvements — Agentic Earthquake Data Pipeline

The current pipeline is a working, end-to-end prototype: four Google ADK
`LlmAgent`s, orchestrated by a `SequentialAgent`, fetch USGS earthquake data,
clean and aggregate it with pandas, gate on data quality, write raw and
processed data to GCS / S3 / local storage in run-partitioned folders, and
publish an HTML report with an LLM-written summary. Below, each area lists
**current state → improvement → why it matters**, specific to this pipeline.

## 1. Scalability

- **Current:** each run re-downloads the full window (≈1,900 events for 30
  days at M2.5+) into one pandas DataFrame. The USGS API rejects queries
  matching more than 20,000 events, so a global M0+ or multi-year query would
  fail.
- **Improvement:** switch to **incremental loads** using the API's
  `updatedafter` parameter (stored as a high-water mark), and upsert into a
  table keyed on event `id` so revised magnitudes overwrite old versions.
  Chunk large windows by week to stay under the 20,000-event cap. Store clean
  data as **partitioned Parquet** (`event_date=YYYY-MM-DD/`) and expose it as a
  **BigQuery external table** (or load it natively). Move heavy transforms to
  BigQuery SQL, or Dataflow/Spark if volume grows by orders of magnitude.
- **Why:** less data transferred per run, correct handling of USGS revisions,
  and queries over months of history without loading everything into memory.

## 2. Orchestration & scheduling

- **Current:** runs on demand (`python main.py` or `adk web`); a new `run_id`
  per run makes artifacts traceable, but nothing runs automatically.
- **Improvement:** containerize `main.py` and run it as a **Cloud Run job**
  triggered by **Cloud Scheduler** (e.g. hourly), or as a task in **Cloud
  Composer/Airflow** if it joins a larger DAG. Make runs **idempotent**:
  derive the partition from the logical query window rather than wall-clock
  time, so a retried run overwrites the same partition instead of duplicating it.
- **ADK roadmap item:** ADK 2.14 deprecates `SequentialAgent` in favour of the
  graph-based **`Workflow`**. Migrating would express the stages as nodes with
  edges, so "halt" becomes a conditional edge (e.g. `quality_status == fail →
  alert node`) instead of a callback, and independent steps (raw and processed
  uploads) can run in parallel. It was deferred because `Workflow` cannot yet
  host `LlmAgent` sub-agents directly and the pinned version runs correctly.

## 3. Error handling & resilience

- **Current:** every tool returns `{"status": "success" | "error"}` and never
  raises; USGS calls retry 3× with backoff and fall back to the summary feed;
  Gemini calls retry 5× on 429/5xx with a 60 s timeout; agents can call
  `halt_pipeline`, and a deterministic guardrail blocks publishing data that
  failed quality checks.
- **Improvement:** **partial-failure recovery** — persist each stage's
  completion in state/storage so a failed run resumes from the last good
  stage instead of re-fetching. Write failed raw payloads to a **dead-letter**
  prefix (`deadletter/earthquakes/...`) for inspection. Add a **deterministic
  fallback path**: if Gemini is unavailable after retries, run the same tool
  sequence without the LLM and use a templated summary, so an LLM outage never
  blocks the data products.
- **Why:** during development a Gemini 503 and a hung request each interrupted
  runs; data delivery should not depend on LLM availability.

## 4. Monitoring & observability

- **Current:** Python `logging` inside tools, a console trace of agent and
  tool events in `main.py`, and the ADK dev UI event trace.
- **Improvement:** emit **structured JSON logs** with `run_id`, agent, tool,
  status, and row counts to **Cloud Logging**; enable ADK's **OpenTelemetry
  tracing** (exported to Cloud Trace) to see per-agent latency and token use.
  Create **log-based metrics and alerts** for zero-row runs, quality `warn`/
  `fail`, runtime above a threshold, and missed schedules (freshness).
- **Why:** today a failure is only visible to whoever ran the command.

## 5. Testing

- **Current:** 25 offline pytest tests on a 13-event fixture covering
  parsing, cleaning, aggregations, KPIs, quality, the tool chain via session
  state, and the local storage backend.
- **Improvement:** integration tests with a **mocked USGS API**
  (`responses`) and **mocked GCS/S3** (`moto`, GCS emulator); a template test
  that renders the HTML and checks for every section; and **ADK agent
  evaluation** with eval sets (`adk eval`) that check tool-call trajectories
  (e.g. "magnitude 5+, past two weeks" → `days=14, min_magnitude=5`) and that
  summaries contain only facts present in the digest. Run everything in
  **GitHub Actions** on every push.
- **Why:** during development the ReportingAgent once attributed a tsunami
  flag to the wrong event; it was fixed by adding per-event facts and an
  accuracy rule, but only an eval set will catch regressions like that
  automatically.

## 6. Data quality

- **Current:** explicit checks (zero rows, magnitude bounds and requested
  minimum, coordinate ranges, duplicates, >25% rows dropped) written to
  `quality_report.json` and shown in the report.
- **Improvement:** declare a schema contract with **Pandera** or **Great
  Expectations** (types, nullability, allowed `mag_type` and `status` values),
  add **freshness** (newest event < N hours old) and **volume anomaly** checks
  (today's count vs. trailing 30-day mean), and track the share of
  `status=automatic` vs. `reviewed` events, since automatic magnitudes are often revised.
- **Why:** catches silent upstream changes (renamed fields, partial feeds)
  that the current range checks would miss.

## 7. Security

- **Current:** no secrets in the repo (`.env`, keys, `output/` gitignored);
  credentials via environment variables and Application Default Credentials;
  the bucket uses uniform access and public-access prevention.
- **Improvement:** keep the Gemini key in **Secret Manager**; run the job as a
  **least-privilege service account** (`roles/storage.objectCreator` on one
  bucket, nothing project-wide); share reports through **short-lived signed
  URLs** rather than public objects; use **Workload Identity Federation** for
  GitHub Actions so CI never holds a JSON key; add secret scanning (e.g.
  gitleaks) as a pre-commit hook.
- **Why:** removes long-lived keys entirely and limits blast radius.

## 8. Cost

- **Current:** about 12 Gemini calls per run on `gemini-3.5-flash-lite`;
  storage is a few MB per run. The free tier of `gemini-3.5-flash` allows only
  ~20 requests/day (about one run), which is why Flash-Lite is the default.
- **Improvement:** cap tokens with `max_output_tokens`; reduce LLM calls by
  letting agents batch independent tool calls (already done for uploads) or by
  making the processing stage a single deterministic tool with one LLM review
  step; add **GCS lifecycle rules** (raw → Nearline after 30 days, delete after
  1 year; delete per-run reports after 90 days, keep `latest/`); use Parquet
  with Snappy/ZSTD compression and drop the duplicate CSV for large tables.
- **Why:** an hourly schedule is ~8,800 runs/year; per-run waste multiplies.

## 9. Visualization

- **Current:** one self-contained HTML page per run (Chart.js + Leaflet),
  plus a `latest/` copy.
- **Improvement:** load processed data to **BigQuery** and build a **Looker
  Studio** dashboard with date-range, region, and magnitude filters and
  history across runs; or host the report on Cloud Run with interactive
  filters; add trend comparisons (this week vs. last).
- **Why:** each HTML file currently shows one run; analysts need history and filtering.

## 10. Agent design

- **Anomaly-detection agent:** compares daily counts and regional activity to
  a historical baseline and explains unusual swarms.
- **Alerting agent:** for any M6+ event or orange/red PAGER alert, posts a
  short briefing to Slack/email.
- **Human-in-the-loop approval:** use ADK tool confirmation so a person must
  approve before publishing when quality status is `warn`.
- **Workflow migration:** move from `SequentialAgent` to ADK `Workflow` (see §2).

## Prioritized roadmap

**Next 2 weeks**
1. Deterministic fallback when Gemini is unavailable, plus a GitHub Actions CI running pytest.
2. Cloud Run job + Cloud Scheduler (daily), with Secret Manager and a least-privilege service account.
3. Structured logging and an alert on zero-row or failed-quality runs.
4. GCS lifecycle rules.

**Next quarter**
1. Incremental `updatedafter` loads into partitioned Parquet + BigQuery; Looker Studio dashboard.
2. ADK eval sets for tool trajectories and summary accuracy.
3. Pandera schema contracts, freshness and volume-anomaly checks.
4. Migrate orchestration to ADK `Workflow`; add anomaly-detection and alerting agents.
