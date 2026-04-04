"""RAG-XPR Streamlit Demo App."""
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG-XPR | Explainable Personality Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subheader { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
    .evidence-card { background: #f1f5f9; border-left: 4px solid #6366f1; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .state-card { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 6px; margin: 8px 0; }
    .prediction-box { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 24px; border-radius: 12px; text-align: center; }
    .metric-box { background: white; border: 1px solid #e2e8f0; padding: 16px; border-radius: 8px; text-align: center; }
    .step-header { color: #6366f1; font-weight: 600; font-size: 1.1rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 RAG-XPR</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Explainable Personality Recognition via Retrieval-Augmented Generation & Chain-of-Personality-Evidence</p>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Sidebar: Configuration
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    llm_provider = st.selectbox("LLM Provider", ["openrouter", "openai", "ollama", "vllm"])
    default_model = (
        "qwen/qwen3.6-plus-preview:free"
        if llm_provider == "openrouter"
        else "gpt-4o-mini" if llm_provider == "openai" else "llama3.1:8b"
    )
    llm_model = st.text_input(
        "Model",
        default_model,
    )
    framework = st.selectbox("Personality Framework", ["mbti", "ocean"])
    num_evidence = st.slider("Max Evidence Sentences", 3, 15, 10)
    num_kb_chunks = st.slider("KB Chunks per Evidence", 1, 10, 5)
    save_intermediate = st.checkbox("Show Intermediate Steps", value=True)

    st.divider()
    st.header("🔑 API Keys")
    api_key = st.text_input("LLM API Key", type="password", placeholder="Enter key for selected provider")
    if api_key:
        os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_MODEL_NAME"] = llm_model

    st.divider()
    st.markdown("""
    **Architecture**
    1. 📖 Evidence Retrieval from text
    2. 🔍 Psychology KB Search (Qdrant)
    3. 🤔 CoPE Reasoning Chain (LLM)
    4. 🎯 Personality Prediction + Explanation
    """)

# ────────────────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📊 Compare Methods", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Input Text")
        sample_texts = {
            "INTJ Example": "I often find myself planning weeks ahead, analyzing every possible scenario. Social events drain me, and I prefer deep one-on-one conversations over small talk. I value efficiency and have little patience for inefficiency.",
            "ENFP Example": "Meeting new people is one of my greatest joys! I love brainstorming ideas and jumping from one exciting project to another. I'm a passionate advocate for causes I believe in.",
            "INFP Example": "I spend a lot of time in my own head, daydreaming and imagining. Authenticity matters deeply to me — I can't stand pretense. I feel others' emotions as if they were my own.",
            "Custom Text": "",
        }
        example_choice = st.selectbox("Load Example", list(sample_texts.keys()))
        default_text = sample_texts[example_choice]
        input_text = st.text_area(
            "Enter text for personality analysis",
            value=default_text,
            height=250,
            placeholder="Enter posts, messages, or essays here...",
        )

        analyze_button = st.button("🔍 Analyze Personality", type="primary", disabled=not input_text.strip())

    with col2:
        st.subheader("Expected MBTI Types")
        st.markdown("""
        | Code | Description |
        |------|-------------|
        | **I/E** | Introversion vs Extraversion |
        | **S/N** | Sensing vs Intuition |
        | **T/F** | Thinking vs Feeling |
        | **J/P** | Judging vs Perceiving |
        """)
        st.info("💡 RAG-XPR grabs evidence from text → maps to psychological states → infers personality")

    if analyze_button and input_text.strip():
        with st.spinner("🔄 Running RAG-XPR pipeline..."):
            try:
                config = {
                    "llm": {
                        "provider": llm_provider,
                        "model": llm_model,
                        "base_url": "https://openrouter.ai/api/v1" if llm_provider == "openrouter" else None,
                        "temperature": 0.1,
                        "max_tokens": 2048,
                    },
                    "cope": {"num_evidence": num_evidence, "num_kb_chunks": num_kb_chunks, "framework": framework, "max_retries_per_step": 2},
                    "evidence_retrieval": {"method": "hybrid", "top_k": num_evidence},
                    "output": {"save_intermediate": save_intermediate},
                }

                from src.rag_pipeline.pipeline import RAGXPRPipeline
                pipeline = RAGXPRPipeline(config)
                result = pipeline.predict(input_text)

                # ── Prediction Display ──────────────────────────────────────
                st.divider()
                col_pred, col_conf = st.columns([1, 2])

                with col_pred:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h1 style="font-size:2.5rem;margin:0">{result.get('predicted_label', 'N/A')}</h1>
                        <p style="margin:0;opacity:0.8">Personality Type ({framework.upper()})</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_conf:
                    details = result.get("prediction_details", {})
                    if framework == "mbti":
                        dims = details.get("dimensions", {})
                        for dim, info in dims.items():
                            confidence = info.get("confidence", 0.5)
                            label = info.get("label", "?")
                            st.progress(confidence, text=f"{dim}: **{label}** ({confidence:.0%})")

                # ── Explanation ──────────────────────────────────────────────
                st.divider()
                st.subheader("📝 Explanation")
                explanation = result.get("explanation", "No explanation generated.")
                st.info(explanation)

                # ── Evidence Chain ───────────────────────────────────────────
                st.subheader("🔗 Evidence Chain")
                evidence_chain = result.get("evidence_chain", [])
                if evidence_chain:
                    for ev in evidence_chain:
                        with st.container():
                            st.markdown(f"""
                            <div class="evidence-card">
                                <strong>📌 Evidence:</strong> "{ev.get('evidence', '')}"<br>
                                <strong>💡 State:</strong> {ev.get('state', '')} &nbsp;|&nbsp;
                                <strong>🎯 Contribution:</strong> {ev.get('trait_contribution', '')}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("No evidence chain available")

                # ── Intermediate Steps ────────────────────────────────────────
                if save_intermediate and "intermediate" in result:
                    with st.expander("🔬 Intermediate Steps (Debug)"):
                        intermediate = result["intermediate"]
                        st.markdown('<p class="step-header">Step 1: Extracted Evidence</p>', unsafe_allow_html=True)
                        for i, ev in enumerate(intermediate.get("step1_evidence", [])):
                            st.markdown(f"**{i+1}.** [{ev.get('behavior_type', '')}] *\"{ev.get('quote', '')}\"* — {ev.get('description', '')}")

                        st.markdown('<p class="step-header">Step 2: Psychological States</p>', unsafe_allow_html=True)
                        for state in intermediate.get("step2_states", []):
                            st.markdown(f"""
                            <div class="state-card">
                                <strong>{state.get('state_label', '')}</strong> (conf: {state.get('confidence', 0):.0%})<br>
                                "{state.get('quote', '')}"<br>
                                <em>{state.get('reasoning', '')}</em>
                            </div>
                            """, unsafe_allow_html=True)

            except ImportError as e:
                st.error(f"Missing dependency: {e}. Please install requirements: `pip install -r requirements.txt`")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                if not any(k in os.environ for k in ("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "VLLM_API_KEY")):
                    st.warning("💡 Set your LLM API key in the sidebar or in the .env file.")

with tab2:
    st.subheader("📊 Compare Methods")
    st.info("Run experiments first, then upload prediction files to compare methods.")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        baseline_file = st.file_uploader("Baseline Predictions (JSONL)", type="jsonl")
    with col_up2:
        rag_xpr_file = st.file_uploader("RAG-XPR Predictions (JSONL)", type="jsonl")

    if baseline_file and rag_xpr_file:
        baseline_preds = [json.loads(line) for line in baseline_file.read().decode().split("\n") if line.strip()]
        ragxpr_preds = [json.loads(line) for line in rag_xpr_file.read().decode().split("\n") if line.strip()]

        from src.evaluation.classification_metrics import \
            compute_classification_metrics
        baseline_y = [p.get("gold_label", "") for p in baseline_preds]
        baseline_pred = [p.get("predicted_label", "") for p in baseline_preds]
        ragxpr_y = [p.get("gold_label", "") for p in ragxpr_preds]
        ragxpr_pred = [p.get("predicted_label", "") for p in ragxpr_preds]

        if baseline_y and ragxpr_y:
            baseline_metrics = compute_classification_metrics(baseline_y, baseline_pred)
            ragxpr_metrics = compute_classification_metrics(ragxpr_y, ragxpr_pred)

            col_b, col_r = st.columns(2)
            with col_b:
                st.metric("Baseline Accuracy", f"{baseline_metrics['accuracy']:.2%}")
                st.metric("Baseline F1 (macro)", f"{baseline_metrics['f1_macro']:.4f}")
            with col_r:
                st.metric("RAG-XPR Accuracy", f"{ragxpr_metrics['accuracy']:.2%}",
                          delta=f"{ragxpr_metrics['accuracy'] - baseline_metrics['accuracy']:+.2%}")
                st.metric("RAG-XPR F1 (macro)", f"{ragxpr_metrics['f1_macro']:.4f}",
                          delta=f"{ragxpr_metrics['f1_macro'] - baseline_metrics['f1_macro']:+.4f}")

with tab3:
    st.subheader("ℹ️ About RAG-XPR")
    st.markdown("""
    **RAG-XPR** (Retrieval-Augmented Generation for Explainable Personality Recognition) combines:

    1. **Retrieval-Augmented Generation (RAG)**: Grounds predictions in a curated psychology knowledge base
    2. **Chain-of-Personality-Evidence (CoPE)**: A 3-step LLM reasoning chain that produces transparent explanations

    ### CoPE Pipeline
    ```
    Input Text
        ↓
    Step 1: Evidence Extraction
        → Identify behavioral quotes that reveal personality traits
        ↓
    Step 2: State Identification (+ KB Retrieval)
        → Map each behavior to a psychological state (grounded in KB)
        ↓
    Step 3: Trait Inference (+ KB Retrieval)
        → Aggregate states → MBTI/OCEAN prediction + natural language explanation
    ```

    ### Datasets
    - **MBTI Kaggle** (~8,600 users, 16-class)
    - **Pandora Reddit** (~10K users, Big Five)
    - **Essays** (2,468 essays, Big Five)
    - **Personality Evd** (dialogues with evidence annotations)

    ### Citation
    > _RAG-based Explainable Personality Recognition (RAG-XPR)_, Master's Thesis, HUST, 2025.
    """)
