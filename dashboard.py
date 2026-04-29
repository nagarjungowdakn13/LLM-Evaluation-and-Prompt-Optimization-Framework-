"""Streamlit dashboard for the LLM Evaluation Framework.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from main import build_runner, load_config, load_dataset
from src.prompts import PromptManager, PromptOptimizer
from src.storage import RunDatabase


# ---------- page config + CSS ----------

st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono&display=swap');

      html, body, [class*="css"] {
        font-family: 'Lora', Georgia, 'Times New Roman', serif;
        color: #3b2e22;
      }
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      h1, h2, h3, h4 { color: #2c2114; letter-spacing: 0.2px; }
      h1 { font-weight: 700; }

      /* Soft hairline divider */
      hr, [data-testid="stSidebar"] hr { border-color: #c9b896; }

      .metric-card {
        background:
          linear-gradient(180deg, #fbf5e6 0%, #f1e7cf 100%);
        padding: 1.0rem 1.2rem;
        border-radius: 6px;
        border: 1px solid #d8c69a;
        border-left: 5px solid #b85c38;       /* terracotta accent */
        box-shadow: 0 1px 0 rgba(60,40,20,0.05), inset 0 0 0 1px rgba(255,255,255,0.4);
        margin-bottom: 0.5rem;
      }
      .metric-label {
        font-size: 0.72rem;
        color: #8a7355;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
      }
      .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #2c2114;
        margin-top: 4px;
        font-family: 'Lora', Georgia, serif;
      }
      .metric-sub { font-size: 0.85rem; color: #6e5a40; margin-top: 4px; font-style: italic; }

      .ungrounded {
        background: rgba(184, 92, 56, 0.18);
        border-bottom: 1px dashed #b85c38;
        padding: 0 3px;
        border-radius: 2px;
        color: #6f2d18;
      }

      .badge-ok  { background:#7a8c4d; color:#faf6ee; padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:600; }
      .badge-bad { background:#a8523a; color:#faf6ee; padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:600; }
      .badge-mid { background:#c89a3a; color:#3b2e22; padding:2px 10px; border-radius:3px; font-size:0.75rem; font-weight:600; }

      .panel {
        background: #fbf5e6;
        padding: 1rem 1.1rem;
        border-radius: 4px;
        border: 1px solid #d8c69a;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.5);
      }
      .mono {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
        white-space: pre-wrap;
        font-size: 0.88rem;
        line-height: 1.55;
        color: #3b2e22;
      }

      /* Tabs — paper tabs */
      [data-baseweb="tab-list"] { border-bottom: 1px solid #c9b896; }
      [data-baseweb="tab"] { font-family: 'Lora', serif; font-weight: 600; }

      /* Code blocks blend with the parchment */
      pre, code { background: #fbf5e6 !important; color: #3b2e22 !important; }

      /* Buttons — terracotta on parchment */
      .stButton button {
        background: #b85c38; color: #faf6ee; border: 1px solid #8d4123;
        font-family: 'Lora', serif; font-weight: 600; letter-spacing: 0.3px;
      }
      .stButton button:hover { background: #a04a2c; border-color: #6f2d18; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- palette ----------

PALETTE = {
    "paper":      "#f7f0e1",
    "paper_dark": "#ece1c9",
    "ink":        "#3b2e22",
    "ink_soft":   "#6e5a40",
    "rule":       "#c9b896",
    "terracotta": "#b85c38",
    "sage":       "#7a8c4d",
    "ochre":      "#c89a3a",
    "wine":       "#7d4e57",
    "denim":      "#5d7a8c",
}

EARTH_SCALE = [
    (0.0, "#a8523a"),   # rust
    (0.5, "#c89a3a"),   # ochre
    (1.0, "#7a8c4d"),   # sage
]

METRIC_COLORS = {
    "correctness":   PALETTE["denim"],
    "consistency":   PALETTE["wine"],
    "hallucination": PALETTE["sage"],
}


def apply_paper_layout(fig, height: int | None = None) -> None:
    fig.update_layout(
        plot_bgcolor=PALETTE["paper"],
        paper_bgcolor=PALETTE["paper"],
        font=dict(color=PALETTE["ink"], family="Lora, Georgia, serif"),
        margin=dict(l=10, r=20, t=10, b=10),
    )
    fig.update_xaxes(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"], linecolor=PALETTE["rule"])
    fig.update_yaxes(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"], linecolor=PALETTE["rule"])
    if height:
        fig.update_layout(height=height)


# ---------- data loading ----------

CONFIG_PATH = "config.yaml"
APP_CONFIG = load_config(CONFIG_PATH)
ACCESS_TOKEN = APP_CONFIG.get("dashboard", {}).get("access_token") or os.getenv("DASHBOARD_ACCESS_TOKEN", "")


def _storage_db() -> RunDatabase:
    storage_cfg = APP_CONFIG.get("storage", {})
    return RunDatabase(storage_cfg.get("sqlite_path", "reports/runs.sqlite3"))


@st.cache_data(show_spinner="Running evaluation against mock LLM…")
def run_evaluation(cache_key: int) -> dict:
    # NOTE: ``cache_key`` is part of the cache key on purpose — incrementing
    # it (from the sidebar "Re-run evaluation" button) invalidates the cache.
    config = APP_CONFIG
    dataset = load_dataset(config["dataset"]["path"])
    pm = PromptManager().load_directory(config["prompts"]["directory"])
    template_names = config["prompts"]["templates"]
    templates = pm.list(template_names)
    runner = build_runner(config, dataset)
    optimizer = PromptOptimizer(evaluate_fn=runner.evaluate)
    comparison = optimizer.compare(templates, dataset)
    ranking = optimizer.rank(comparison)
    rendered_prompts = {t.name: {ex["id"]: t.render(ex) for ex in dataset} for t in templates}
    return {
        "config": config,
        "dataset": {ex["id"]: ex for ex in dataset},
        "comparison": comparison,
        "ranking": ranking,
        "templates": template_names,
        "prompts": rendered_prompts,
    }


def run_evaluation_streaming(cache_key: int) -> dict:
    del cache_key
    config = APP_CONFIG
    dataset = load_dataset(config["dataset"]["path"])
    pm = PromptManager().load_directory(config["prompts"]["directory"])
    template_names = config["prompts"]["templates"]
    templates = pm.list(template_names)
    runner = build_runner(config, dataset)

    progress = st.progress(0)
    status = st.empty()
    live_output = st.empty()
    comparison: dict[str, dict] = {}
    rendered_prompts = {t.name: {ex["id"]: t.render(ex) for ex in dataset} for t in templates}
    completed = 0
    total = max(1, len(templates) * len(dataset))

    for template in templates:
        per_example = []
        scores = []
        sub_scores: dict[str, list[float]] = {
            "correctness": [],
            "consistency": [],
            "hallucination": [],
        }
        for example in dataset:
            status.caption(f"Running {template.name} on {example['id']} ({completed + 1}/{total})")
            buffer: list[str] = []

            def on_chunk(chunk: str, _buffer=buffer) -> None:
                _buffer.append(chunk)
                live_output.code("".join(_buffer), language="text")

            result = runner.evaluate(template, example, stream_callback=on_chunk)
            per_example.append(result.to_dict())
            scores.append(result.overall)
            sub_scores["correctness"].append(result.correctness["overall"])
            sub_scores["consistency"].append(result.consistency["score"])
            sub_scores["hallucination"].append(result.hallucination["combined_score"])
            completed += 1
            progress.progress(int((completed / total) * 100))

        comparison[template.name] = {
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "average_correctness": sum(sub_scores["correctness"]) / len(sub_scores["correctness"]) if scores else 0.0,
            "average_consistency": sum(sub_scores["consistency"]) / len(sub_scores["consistency"]) if scores else 0.0,
            "average_hallucination": sum(sub_scores["hallucination"]) / len(sub_scores["hallucination"]) if scores else 0.0,
            "examples": per_example,
        }

    ranking = PromptOptimizer(evaluate_fn=runner.evaluate).rank(comparison)
    state = {
        "config": config,
        "dataset": {ex["id"]: ex for ex in dataset},
        "comparison": comparison,
        "ranking": ranking,
        "templates": template_names,
        "prompts": rendered_prompts,
    }
    if config.get("storage", {}).get("enabled", True):
        db = _storage_db()
        run_id = db.save_run(
            comparison=comparison,
            ranking=ranking,
            config=config,
            report_paths={},
        )
        state["run_id"] = run_id
    progress.progress(100)
    status.caption("Streaming run complete")
    return state


def load_report_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "comparison": payload["comparison"],
        "ranking": [tuple(r) for r in payload["ranking"]],
        "templates": list(payload["comparison"].keys()),
        "dataset": None,
        "prompts": None,
        "config": None,
    }


def load_saved_run(run_id: str) -> dict:
    record = _storage_db().load_run(run_id)
    if record is None:
        raise FileNotFoundError(f"No saved run found for {run_id}")
    return {
        "comparison": record["comparison"],
        "ranking": record["ranking"],
        "templates": list(record["comparison"].keys()),
        "dataset": None,
        "prompts": None,
        "config": record["config"],
        "run_id": record["run_id"],
    }


# ---------- helpers ----------

def score_class(score: float) -> str:
    if score >= 0.75:
        return "badge-ok"
    if score >= 0.5:
        return "badge-mid"
    return "badge-bad"


def score_color(score: float) -> str:
    if score >= 0.75:
        return PALETTE["sage"]
    if score >= 0.5:
        return PALETTE["ochre"]
    return PALETTE["terracotta"]


def metric_card(label: str, value: str, sub: str = "", color: str = "#b85c38") -> str:
    return (
        f'<div class="metric-card" style="border-left-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f"</div>"
    )


def highlight_ungrounded(text: str, ungrounded: list[str]) -> str:
    safe = escape(text)
    if not ungrounded:
        return f'<div class="mono panel">{safe}</div>'
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(set(ungrounded), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    highlighted = pattern.sub(lambda m: f'<span class="ungrounded">{m.group(0)}</span>', safe)
    return f'<div class="mono panel">{highlighted}</div>'


# ---------- sidebar ----------

st.sidebar.title("LLM Eval Dashboard")
st.sidebar.caption("Evaluation + prompt-optimization framework")

if ACCESS_TOKEN:
    if st.session_state.get("dashboard_authenticated") is not True:
        entered = st.sidebar.text_input("Access token", type="password")
        if st.sidebar.button("Unlock dashboard") and entered == ACCESS_TOKEN:
            st.session_state.dashboard_authenticated = True
        if st.session_state.get("dashboard_authenticated") is not True:
            st.info("Enter the dashboard access token to continue.")
            st.stop()

mode = st.sidebar.radio(
    "Data source",
    ["Run live (streaming)", "Load saved report", "Load saved run"],
    help="Live mode runs the full framework with streamed output. Saved modes load from reports/ or SQLite.",
)

if mode == "Run live (streaming)":
    if "run_token" not in st.session_state:
        st.session_state.run_token = 0
    if st.sidebar.button("Re-run evaluation", width='stretch'):
        st.session_state.run_token += 1
    if st.session_state.get("live_run_token") != st.session_state.run_token or "live_state" not in st.session_state:
        state = run_evaluation_streaming(st.session_state.run_token)
        st.session_state.live_state = state
        st.session_state.live_run_token = st.session_state.run_token
    else:
        state = st.session_state.live_state
else:
    if mode == "Load saved report":
        reports = sorted(Path("reports").glob("evaluation_*.json"), reverse=True)
        if not reports:
            st.sidebar.error("No reports under reports/. Run `python main.py run` first.")
            st.stop()
        chosen = st.sidebar.selectbox("Report file", reports, format_func=lambda p: p.name)
        state = load_report_file(chosen)
    else:
        runs = _storage_db().list_runs(limit=50)
        if not runs:
            st.sidebar.error("No saved runs found in SQLite yet. Run the evaluator first.")
            st.stop()
        run_labels = {
            run["run_id"]: f"{run['created_at']} | {run['provider']} | {run['model']} | {run['summary']['best_template']}"
            for run in runs
        }
        chosen_run = st.sidebar.selectbox("Saved run", list(run_labels), format_func=lambda run_id: run_labels[run_id])
        state = load_saved_run(chosen_run)

st.sidebar.divider()
st.sidebar.markdown(
    "**Quick reference**\n\n"
    "- **Correctness** = exact + semantic + schema + rules\n"
    "- **Consistency** = pairwise similarity over N runs\n"
    "- **Hallucination** = grounding ratio × consistency"
)


# ---------- header KPIs ----------

comparison = state["comparison"]
ranking = state["ranking"]
templates = state["templates"]

st.title("LLM Evaluation & Prompt Optimization")
st.caption(
    "Each prompt template is scored against the same evaluation set on correctness, "
    "consistency, hallucination, schema validity and rule checks. Higher overall = safer prompt."
)

best_name, best_score = ranking[0]
worst_name, worst_score = ranking[-1]
lift = (best_score - worst_score) / worst_score if worst_score else float("inf")
total_examples = len(next(iter(comparison.values()))["examples"]) if comparison else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(metric_card("Best template", best_name, f"score {best_score:.3f}", score_color(best_score)), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Worst template", worst_name, f"score {worst_score:.3f}", score_color(worst_score)), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("Lift (best vs worst)", f"{lift*100:.0f}%", "improvement from prompt design", PALETTE["wine"]), unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("Examples evaluated", str(total_examples), f"{len(templates)} templates × {total_examples} = {len(templates)*total_examples} runs", PALETTE["denim"]), unsafe_allow_html=True)


# ---------- main tabs ----------

tab_lead, tab_breakdown, tab_heat, tab_drill, tab_failures, tab_runs = st.tabs(
    [
        "Leaderboard",
        "Per-template breakdown",
        "Heatmap",
        "Drill-down inspector",
        "Failure analysis",
        "Run comparison",
    ]
)


# ---- Leaderboard ----
with tab_lead:
    st.subheader("Prompt ranking")
    df_rank = pd.DataFrame(
        [
            {
                "template": name,
                "overall": comparison[name]["average_score"],
                "correctness": comparison[name]["average_correctness"],
                "consistency": comparison[name]["average_consistency"],
                "hallucination": comparison[name]["average_hallucination"],
            }
            for name, _ in ranking
        ]
    )
    fig = px.bar(
        df_rank.sort_values("overall"),
        x="overall",
        y="template",
        orientation="h",
        color="overall",
        color_continuous_scale=EARTH_SCALE,
        range_color=(0, 1),
        text=df_rank.sort_values("overall")["overall"].map(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(color=PALETTE["ink"], family="Lora, serif"),
        marker_line=dict(color="#8d4123", width=0.5),
    )
    apply_paper_layout(fig, height=80 + 60 * len(df_rank))
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(range=[0, 1.05], title="Overall score (0–1)"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Sub-metric averages")
    melted = df_rank.melt(
        id_vars="template",
        value_vars=["correctness", "consistency", "hallucination"],
        var_name="metric",
        value_name="score",
    )
    fig2 = px.bar(
        melted, x="template", y="score", color="metric", barmode="group",
        color_discrete_map=METRIC_COLORS,
    )
    fig2.update_traces(marker_line=dict(color=PALETTE["ink"], width=0.4))
    apply_paper_layout(fig2)
    fig2.update_layout(
        yaxis=dict(range=[0, 1], title="Average score"),
        xaxis_title="",
        legend_title_text="",
    )
    st.plotly_chart(fig2, width='stretch')


# ---- Per-template breakdown ----
with tab_breakdown:
    st.subheader("Strengths & weaknesses (radar)")
    radar = go.Figure()
    metrics = ["correctness", "consistency", "hallucination"]
    radar_colors = [PALETTE["terracotta"], PALETTE["wine"], PALETTE["sage"], PALETTE["denim"], PALETTE["ochre"]]
    for i, name in enumerate(templates):
        values = [
            comparison[name]["average_correctness"],
            comparison[name]["average_consistency"],
            comparison[name]["average_hallucination"],
        ]
        color = radar_colors[i % len(radar_colors)]
        radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=name,
                line=dict(color=color, width=2),
                fillcolor=color,
                opacity=0.35,
            )
        )
    radar.update_layout(
        polar=dict(
            bgcolor=PALETTE["paper"],
            radialaxis=dict(range=[0, 1], showline=False, gridcolor=PALETTE["rule"], tickfont=dict(color=PALETTE["ink_soft"])),
            angularaxis=dict(gridcolor=PALETTE["rule"], tickfont=dict(color=PALETTE["ink"])),
        ),
        paper_bgcolor=PALETTE["paper"],
        font=dict(color=PALETTE["ink"], family="Lora, serif"),
        height=460,
        showlegend=True,
        legend=dict(bgcolor=PALETTE["paper_dark"], bordercolor=PALETTE["rule"], borderwidth=1),
    )
    st.plotly_chart(radar, width='stretch')

    st.subheader("Per-template summary")
    for name in templates:
        d = comparison[name]
        with st.expander(f"{name}  —  overall {d['average_score']:.3f}", expanded=False):
            cA, cB, cC = st.columns(3)
            cA.metric("Correctness", f"{d['average_correctness']:.3f}")
            cB.metric("Consistency", f"{d['average_consistency']:.3f}")
            cC.metric("Hallucination", f"{d['average_hallucination']:.3f}")
            rows = []
            for ex in d["examples"]:
                schema = ex["correctness"].get("schema") or {}
                schema_valid = schema.get("valid")
                grounding_class = ex["hallucination"].get("classification") or "—"
                error_tags = ", ".join(
                    sorted({(ev.get("category") or "?") for ev in ex.get("error_events", []) or []})
                ) or "—"
                rows.append(
                    {
                        "example": ex["example_id"],
                        "overall": round(ex["overall"], 3),
                        "exact": round(ex["correctness"]["exact_match"], 2),
                        "semantic": round(ex["correctness"]["semantic_similarity"], 2),
                        "schema": "OK" if schema_valid else ("—" if schema_valid is None else "FAIL"),
                        "rules": round(ex["correctness"].get("rule_based", {}).get("score", float("nan")), 2),
                        "halluc.": round(ex["hallucination"]["combined_score"], 2),
                        "class": grounding_class,
                        "errors": error_tags,
                        "ungrounded": ", ".join(ex["hallucination"]["ungrounded_terms"][:5]) or "—",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ---- Heatmap ----
with tab_heat:
    st.subheader("Score heatmap (templates × examples)")
    example_ids = [ex["example_id"] for ex in next(iter(comparison.values()))["examples"]]
    matrix = []
    for name in templates:
        row = [next(e["overall"] for e in comparison[name]["examples"] if e["example_id"] == eid) for eid in example_ids]
        matrix.append(row)
    heat = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=example_ids,
            y=templates,
            colorscale=EARTH_SCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="overall",
                tickfont=dict(color=PALETTE["ink"]),
                outlinecolor=PALETTE["rule"],
                outlinewidth=1,
            ),
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"color": PALETTE["ink"], "family": "JetBrains Mono, monospace"},
            xgap=2,
            ygap=2,
        )
    )
    apply_paper_layout(heat, height=120 + 60 * len(templates))
    heat.update_layout(
        xaxis=dict(title="Example"),
        yaxis=dict(title="Template", autorange="reversed"),
    )
    st.plotly_chart(heat, width='stretch')
    st.caption(
        "Cells are the per-example overall score. Red columns identify hard examples; "
        "red rows identify under-performing prompts."
    )


# ---- Drill-down ----
with tab_drill:
    st.subheader("Inspect a single (template, example) pair")
    col_t, col_e = st.columns(2)
    template_choice = col_t.selectbox("Template", templates, index=0)
    example_ids = [ex["example_id"] for ex in comparison[template_choice]["examples"]]
    example_choice = col_e.selectbox("Example", example_ids, index=0)

    record = next(e for e in comparison[template_choice]["examples"] if e["example_id"] == example_choice)
    overall = record["overall"]
    schema = record["correctness"].get("schema") or {}
    schema_valid = schema.get("valid")
    schema_errors = schema.get("errors", [])

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Overall", f"{overall:.3f}", "weighted blend", score_color(overall)), unsafe_allow_html=True)
    k2.markdown(
        metric_card(
            "Schema",
            "valid" if schema_valid else ("n/a" if schema_valid is None else "INVALID"),
            "JSON-Schema check",
            PALETTE["sage"] if schema_valid else (PALETTE["ink_soft"] if schema_valid is None else PALETTE["terracotta"]),
        ),
        unsafe_allow_html=True,
    )
    k3.markdown(
        metric_card(
            "Hallucination",
            f"{record['hallucination']['combined_score']:.2f}",
            "grounding × consistency",
            score_color(record["hallucination"]["combined_score"]),
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        metric_card(
            "Consistency",
            f"{record['consistency']['score']:.2f}",
            f"over {record['consistency']['n_runs']} runs",
            score_color(record["consistency"]["score"]),
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Model output (ungrounded terms highlighted)")
    st.markdown(
        highlight_ungrounded(record["output"], record["hallucination"]["ungrounded_terms"]),
        unsafe_allow_html=True,
    )

    st.markdown("### Expected output")
    st.code(json.dumps(record["expected"], indent=2, ensure_ascii=False), language="json")

    if state.get("dataset"):
        ex_full = state["dataset"].get(example_choice)
        if ex_full and ex_full.get("context"):
            st.markdown("### Grounding context")
            st.info(ex_full["context"])

    if state.get("prompts"):
        with st.expander("Rendered prompt sent to the LLM"):
            st.code(state["prompts"][template_choice][example_choice], language="text")

    if schema_errors:
        st.markdown("### Schema errors")
        for e in schema_errors:
            st.error(e)

    rb = record["correctness"].get("rule_based")
    if rb:
        st.markdown("### Rule checks")
        cols = st.columns(max(1, len(rb["passed"]) + len(rb["failed"])))
        i = 0
        for name in rb["passed"]:
            cols[i].success(f"PASS  {name}")
            i += 1
        for name in rb["failed"]:
            cols[i].error(f"FAIL  {name}")
            i += 1

    st.markdown("### Consistency runs")
    runs = record["consistency"].get("outputs", [])
    if runs:
        run_cols = st.columns(len(runs))
        for i, out in enumerate(runs):
            with run_cols[i]:
                st.caption(f"Run {i+1}")
                st.code(out, language="text")

    sims = record["consistency"].get("pairwise_similarities", [])
    if sims:
        st.caption(
            "Pairwise similarities between runs: "
            + ", ".join(f"{s:.2f}" for s in sims)
        )


# ---- Failure analysis ----
with tab_failures:
    st.subheader("Where each prompt fails")
    st.caption(
        "Errors are tagged from the structured taxonomy: hallucination, "
        "schema violation, formatting error, missing information, incorrect reasoning. "
        "A single failing example may carry more than one tag."
    )

    rows = []
    for name in templates:
        d = comparison[name]
        counts = d.get("error_counts") or {}
        if not counts and "examples" in d:
            counter = {}
            for ex in d["examples"]:
                for ev in ex.get("error_events", []) or []:
                    counter[ev.get("category", "unknown")] = counter.get(ev.get("category", "unknown"), 0) + 1
            counts = counter
        for category, n in counts.items():
            rows.append({"template": name, "category": category, "count": n})

    if not rows:
        st.success("No errors detected on any template — all examples passed every check.")
    else:
        err_df = pd.DataFrame(rows)
        fig_err = px.bar(
            err_df,
            x="template",
            y="count",
            color="category",
            barmode="stack",
            color_discrete_map={
                "hallucination":       PALETTE["terracotta"],
                "schema_violation":    PALETTE["wine"],
                "formatting_error":    PALETTE["ochre"],
                "missing_information": PALETTE["denim"],
                "incorrect_reasoning": PALETTE["sage"],
            },
        )
        fig_err.update_traces(marker_line=dict(color=PALETTE["ink"], width=0.4))
        apply_paper_layout(fig_err)
        fig_err.update_layout(
            yaxis=dict(title="Failure events"),
            xaxis_title="",
            legend_title_text="Category",
        )
        st.plotly_chart(fig_err, width="stretch")

    st.subheader("Grounding classification")
    st.caption(
        "Each output is classified as grounded, partially grounded, or "
        "hallucinated based on grounding ratio + missing-entity recall + "
        "unsupported-claim density."
    )
    cls_rows = []
    for name in templates:
        cls = comparison[name].get("grounding_classes") or {}
        if not cls and "examples" in comparison[name]:
            counter = {}
            for ex in comparison[name]["examples"]:
                gc = ex.get("hallucination", {}).get("classification")
                if gc:
                    counter[gc] = counter.get(gc, 0) + 1
            cls = counter
        for klass, n in cls.items():
            cls_rows.append({"template": name, "class": klass, "count": n})
    if cls_rows:
        cls_df = pd.DataFrame(cls_rows)
        fig_cls = px.bar(
            cls_df,
            x="template",
            y="count",
            color="class",
            barmode="stack",
            color_discrete_map={
                "grounded":             PALETTE["sage"],
                "partially_grounded":   PALETTE["ochre"],
                "hallucinated":         PALETTE["terracotta"],
            },
        )
        apply_paper_layout(fig_cls)
        fig_cls.update_layout(
            yaxis=dict(title="Examples"), xaxis_title="", legend_title_text="",
        )
        st.plotly_chart(fig_cls, width="stretch")
    else:
        st.info("No grounding classifications available in this comparison.")


# ---- Run comparison (cross-run analytics) ----
with tab_runs:
    st.subheader("Across runs (SQLite history)")
    st.caption(
        "Each batch evaluation is stored in reports/runs.sqlite3. "
        "This view ranks every (run, template) pair — useful for tracking "
        "improvements across iterations of your prompt set."
    )
    try:
        from src.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(sqlite_path="reports/runs.sqlite3")
        leaderboard = tracker.leaderboard(limit=10)
        rows_lb = [
            {
                "run_id":       r["run_id"],
                "created_at":   r["created_at"],
                "model":        r.get("model"),
                "best_template": (r.get("summary") or {}).get("best_template"),
                "best_score":   round((r.get("summary") or {}).get("best_score") or 0.0, 3),
                "templates":    (r.get("summary") or {}).get("template_count"),
            }
            for r in leaderboard
        ]
        if rows_lb:
            st.dataframe(pd.DataFrame(rows_lb), width="stretch", hide_index=True)
        else:
            st.info("No saved runs yet. Run `python main.py run` to populate the database.")

        st.subheader("Best (run, template) across history")
        cross = tracker.best_templates_across_runs(limit=20)
        if cross:
            cross_df = pd.DataFrame(cross[:25])
            cross_df["score"] = cross_df["score"].round(3)
            st.dataframe(cross_df, width="stretch", hide_index=True)

        if len(leaderboard) >= 2:
            st.subheader("Compare two runs")
            ids = [r["run_id"] for r in leaderboard]
            colA, colB = st.columns(2)
            run_a = colA.selectbox("Run A (baseline)", ids, index=min(1, len(ids) - 1))
            run_b = colB.selectbox("Run B (new)", ids, index=0)
            if run_a != run_b:
                delta = tracker.compare_runs(run_a, run_b)
                st.metric(
                    "Best-score delta (B − A)",
                    f"{delta.score_delta:+.3f}",
                    delta_color="normal" if delta.score_delta >= 0 else "inverse",
                )
                tdf = pd.DataFrame(
                    [
                        {"template": k, "delta": round(v, 3)}
                        for k, v in delta.template_changes.items()
                    ]
                )
                st.dataframe(tdf, width="stretch", hide_index=True)
    except Exception as exc:  # pragma: no cover
        st.warning(f"Run history unavailable: {exc}")
