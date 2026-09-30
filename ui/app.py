"""Streamlit demo: ask by text or voice, watch the agent's steps live, get verified snippets.

Run: .venv/bin/streamlit run ui/app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace  # noqa: E402

from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import JavascriptLexer  # noqa: E402

from src import config  # noqa: E402
from src.config import AgentBudget  # noqa: E402

EXAMPLES = [
    "Where is the Bluetooth-settings deeplink used?",
    "Which files call tool setValue before tool emit?",
    "Where is EVENTS.DEVICE.NEW_STATE emitted?",
    "Who calls purgeStates?",
    "What does BluetoothManager.scan call?",
    "How does Gladys purge old device states?",
    "Which MCP tool turns devices on or off?",
]

st.set_page_config(page_title="Code Intelligence Agent", page_icon="🔎", layout="wide")

CSS = """
<style>
.snippet {border: 1px solid rgba(128,128,128,.35); border-radius: 8px; overflow-x: auto;
          font-size: 0.82rem; margin-bottom: .6rem;}
.snippet pre {margin: 0; padding: .5rem .75rem; line-height: 1.35;}
.snippet .hll {background-color: rgba(255, 213, 79, .35); display: block;}
.snippet .linenos {color: #888; padding-right: .6rem; user-select: none;}
.loc {font-family: ui-monospace, monospace; font-weight: 600;}
.badge {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .78rem;
        margin-right: 6px; border: 1px solid rgba(128,128,128,.5);}
.high {background: rgba(76,175,80,.18);} .medium {background: rgba(255,193,7,.2);}
.low {background: rgba(244,67,54,.18);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f"<style>{HtmlFormatter(style='friendly').get_style_defs('.snippet')}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading index and models…")
def _warm() -> dict:
    """Open the index once per server process and warm the retriever/embedding model."""
    from src.agent.tools import code_tools
    from src.retrieval.search import get_retriever

    code_tools.get_context()
    r = get_retriever()
    try:
        r.search("warm up", 3)
    except Exception as exc:  # noqa: BLE001
        return {"warm_error": str(exc)}
    return {}


def _manifest() -> dict:
    try:
        return json.loads(config.MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _code_html(sn: dict) -> str:
    fmt = HtmlFormatter(
        linenos="inline",
        linenostart=sn["start_line"],
        hl_lines=[h - sn["start_line"] + 1 for h in sn["highlight_lines"]],
        nowrap=False,
    )
    return f'<div class="snippet">{highlight(sn["text"], JavascriptLexer(), fmt)}</div>'


def _dot(paths: list[list[str]]) -> str:
    edges, nodes = set(), set()
    for p in paths[:30]:
        clean = [n.split(" → ")[0][:48] for n in p]
        for a, b in zip(clean, clean[1:], strict=False):
            edges.add((a, b))
            nodes.update((a, b))
    lines = [
        "digraph G {",
        'rankdir=LR; node [shape=box, style="rounded,filled", '
        'fillcolor="#eef3fb", fontname="Helvetica", fontsize=10]; edge [color="#7a8aa6"];',
    ]
    for a, b in sorted(edges):
        lines.append(f'"{html.escape(a)}" -> "{html.escape(b)}";')
    lines.append("}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle(
        "Use local LLM (Ollama)",
        value=True,
        help="Off = deterministic graph/catalog answers + ranked retrieval only.",
    )
    explain = st.toggle(
        "Stream an LLM explanation after the results (optional)",
        value=config.EXPLAIN_DEFAULT,
        disabled=not use_llm,
    )
    optimize = st.toggle("Show optimisation suggestions", value=False)
    graph_boost = st.toggle("Graph/catalog boost in retrieval", value=True)
    rerank = st.toggle("Cross-encoder rerank (slower)", value=False)
    refine_on = st.toggle(
        "Refine when confidence is low (graph expansion + constrained LLM rewrite)",
        value=True,
        help="≤ 2 grounded refinement rounds: call-graph/catalog expansion (deterministic), "
        "then an LLM that may only pick real index terms. First results never wait for the LLM.",
    )
    st.divider()
    from src.agent.llm import get_llm

    llm = get_llm()
    up = llm.available()
    st.markdown(
        f"**LLM:** `{llm.model}` {'🟢 up' if up else '🔴 down'}  \n"
        f"fallback `{llm.fallback_model}` · num_gpu={config.LLM_NUM_GPU} (CPU)"
    )
    m = _manifest()
    if m:
        c = m.get("counts", {})
        st.markdown(
            f"**Index:** Gladys @ `{str(m.get('repo_commit'))[:7]}`  \n"
            f"{m.get('files_indexed')} files · {c.get('symbols')} symbols · {c.get('calls')} calls"
            f" · {c.get('chunks')} chunks · {c.get('catalog')} catalog rows  \n"
            f"embeddings `{m.get('embedding_model')}`"
        )
    else:
        st.error("No index found. Run `./run.sh index` first.")

# ------------------------------------------------------------------------------------ main
st.title("🔎 Agentic Code Intelligence")
st.caption(
    "Ask about the Gladys Assistant JavaScript codebase. Structural and usage questions "
    "are answered from the AST, call graph and domain catalog; every file:line shown is "
    "re-read from disk. Everything runs locally on CPU."
)

warm = _warm()
if warm.get("warm_error"):
    st.warning(f"Retriever warm-up failed: {warm['warm_error']}")

if "query" not in st.session_state:
    st.session_state.query = EXAMPLES[0]

cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 4].button(ex, key=f"ex{i}", width="stretch"):
        st.session_state.query = ex
        st.session_state.run = True

audio = st.audio_input("🎙️ Ask by voice (transcribed locally with faster-whisper)")
if audio is not None:
    digest = hash(audio.getvalue())
    if st.session_state.get("audio_digest") != digest:
        st.session_state.audio_digest = digest
        with st.spinner("Transcribing on CPU…"):
            try:
                from src.speech.stt import transcribe

                text, ms = transcribe(audio.getvalue())
                if text:
                    st.session_state.query = text
                    st.session_state.run = True
                    st.success(f"Heard: “{text}” ({ms / 1000:.1f} s)")
                else:
                    st.warning("No speech detected.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Speech-to-text failed: {exc}")

with st.form("ask"):
    query = st.text_input("Question", key="query")
    submitted = st.form_submit_button("Ask", type="primary")

STEP_ICONS = {
    "plan": "🧭",
    "search": "🔎",
    "read": "📖",
    "assess": "📏",
    "refine": "🔁",
    "verify": "✅",
    "stop": "⏹️",
    "llm": "🤖",
}


def _step_line(step) -> str:
    ins = f" `{json.dumps(step.inputs)[:120]}`" if step.inputs else ""
    ms = f" · {step.elapsed_ms:.0f} ms" if step.elapsed_ms else ""
    return (
        f"{STEP_ICONS.get(step.kind, '•')} **{step.kind}: {step.name}**{ins}{ms}  \n"
        f"<small>{html.escape(step.summary[:220])}</small>"
    )


def _render_results(d: dict) -> None:
    top = st.columns([3, 1, 1, 1])
    top[0].markdown(
        f'<span class="badge {d["confidence"]}">confidence: {d["confidence"]}</span>'
        f'<span class="badge">route: {d["route"].get("type")}</span>'
        f'<span class="badge">refinement rounds: {d["refinement_rounds"]}</span>'
        + (
            '<span class="badge medium">updated by LLM refinement</span>'
            if d["results_changed_by_llm"]
            else ""
        ),
        unsafe_allow_html=True,
    )
    top[1].metric("First results", f"{d['latency_ms'] / 1000:.2f} s")
    top[2].metric("Final results", f"{d['final_latency_ms'] / 1000:.2f} s")
    top[3].metric("Steps", len(d["steps"]))
    st.subheader("Answer")
    st.markdown(d["answer"] or "_No answer._")
    if d["route"].get("rewritten_query"):
        st.caption(
            f"Low confidence → searched again with index terms picked by the LLM: "
            f"“{d['route']['rewritten_query']}”"
        )
    if d["error"]:
        st.error(d["error"])
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Verified snippets ({len(d['snippets'])})")
        for sn in d["snippets"]:
            st.markdown(
                f'<div class="loc">{html.escape(sn["location"])}'
                f"{' · ' + html.escape(sn['symbol']) if sn['symbol'] else ''}</div>"
                f"<small>{html.escape(sn['reason'])}</small>",
                unsafe_allow_html=True,
            )
            st.markdown(_code_html(sn), unsafe_allow_html=True)
    with right:
        if d["plan"].get("subplans"):
            st.subheader("Plan")
            for sp in d["plan"]["subplans"]:
                st.markdown(
                    f"**{sp['route']['type']}** — {html.escape(sp['text'])}  \n"
                    + " → ".join(sp["steps"])
                )
        if d["call_paths"]:
            st.subheader("Call paths")
            st.graphviz_chart(_dot(d["call_paths"]), width="stretch")
        st.subheader("Locations")
        st.code("\n".join(d["locations"]) or "—", language=None)
        if d["optimizations"]:
            st.subheader(f"Optimisation suggestions ({len(d['optimizations'])})")
            st.caption("Static checks on the surfaced code; advisory, not verified by execution.")
            for o in d["optimizations"]:
                with st.expander(f"{o['check']} · {o['location']}"):
                    st.write(o.get("explanation") or o["message"])
                    st.code(o["evidence"], language="javascript")
        with st.expander("Full step trace (structured events)"):
            st.json(d["steps"])


if submitted or st.session_state.pop("run", False):
    from src.agent.agent import answer, continue_with_llm, stream_explanation

    ranking = replace(config.AGENT_RANKING, use_graph_boost=graph_boost, use_rerank=rerank)
    budget = AgentBudget(refine_rounds=2 if refine_on else 0)
    refine_cfg = config.REFINE if refine_on else replace(config.REFINE, expand=False, rewrite=False)
    status = st.status("Planning and searching…", expanded=True)
    res = answer(
        query,
        use_llm=use_llm,
        optimize=optimize,
        ranking=ranking,
        budget=budget,
        refine=refine_cfg,
        on_step=lambda s: status.markdown(_step_line(s), unsafe_allow_html=True),
    )
    status.update(
        label=f"First results in {res.latency_ms / 1000:.2f} s · {len(res.steps)} steps",
        state="complete" if not res.error else "error",
        expanded=False,
    )
    explanation_box = st.empty()
    results_box = st.empty()
    with results_box.container():
        _render_results(res.to_dict())
    # Phase B: grounded LLM refinement (planning fallback / constrained rewrite) — after the
    # first results are already on screen.
    if use_llm and refine_on and not res.error:
        status_b = st.status(
            "Refining with the local LLM (terms picked from the index only)…", expanded=False
        )
        changed = continue_with_llm(
            res, on_step=lambda s: status_b.markdown(_step_line(s), unsafe_allow_html=True)
        )
        status_b.update(
            label="Results updated by refinement"
            if changed
            else "Refinement confirmed the first results",
            state="complete",
        )
        if changed:
            with results_box.container():
                _render_results(res.to_dict())
    if explain and use_llm and res.snippets:
        with explanation_box.container():
            st.markdown(
                "**Explanation** <small>(local LLM, from the verified snippets)</small>",
                unsafe_allow_html=True,
            )
            st.write_stream(stream_explanation(res))
