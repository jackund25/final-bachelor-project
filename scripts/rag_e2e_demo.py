"""End-to-end test: RF prediction → PatientState → conditioned RAG → Gemini advisory.

Run:
    python test_rag_e2e.py

Requires GOOGLE_API_KEY in environment or .env file.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on shell env vars


def _hr(title: str = "") -> None:
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def test_patient_state_module():
    _hr("STEP 1 — PatientState module")
    from src.patient_state import PatientState

    # Scenario: hyperglycemia rising rapidly
    state = PatientState.from_model_output(
        patient_id="ohio_559",
        current_glucose=180.0,
        predicted_glucose=210.0,
        feature_row={"insulin": 0.0, "carbs": 60.0, "activity": 0, "stress": 8},
    )

    print(f"  Patient ID       : {state.patient_id}")
    print(f"  Current glucose  : {state.current_glucose:.1f} mg/dL")
    print(f"  Predicted glucose: {state.predicted_glucose:.1f} mg/dL")
    print(f"  Delta            : {state.glucose_delta:+.1f} mg/dL")
    print(f"  Trend            : {state.trend_direction} ({state.trend_rate})")
    print(f"  Risk level       : {state.risk_level}")
    print(f"  Risk label       : {state.risk_label}")
    print(f"  Urgency          : {state.urgency}")
    print(f"  ✓ PatientState OK")
    return state


def test_conditioned_query(state):
    _hr("STEP 2 — PredictionConditionedQueryBuilder")
    from src.rag.conditioned_query import PredictionConditionedQueryBuilder, QueryStrategy

    builder = PredictionConditionedQueryBuilder(strategy=QueryStrategy.COMPREHENSIVE)
    cq = builder.build(state)

    print("  Primary query (sent to ChromaDB retriever):")
    print(f"  >>> {cq.primary_query}")
    print()
    print("  LLM context (injected into Gemini prompt):")
    for line in cq.llm_context.split("\n"):
        print(f"  {line}")
    print()
    print(f"  ConditionedQuery repr: {repr(cq)}")
    print(f"  Pipeline kwargs keys : {list(cq.to_pipeline_kwargs().keys())}")
    print(f"  ✓ ConditionedQuery OK")
    return cq


def test_pipeline_e2e(cq):
    _hr("STEP 3 — RAGPipeline end-to-end (Gemini + sentence-transformers)")
    from src.rag.pipeline import RAGPipeline

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("  ⚠  GOOGLE_API_KEY not set — testing with template provider instead.")
        print("     Set GOOGLE_API_KEY in .env to enable real Gemini calls.")
        provider = "template"
    else:
        provider = "gemini"
        print(f"  LLM provider: {provider} (key set ✓)")

    print("  Building pipeline...")
    pipeline = RAGPipeline(
        llm_provider=provider,
        embed_provider="sentence-transformers",
    )
    pipeline.build()
    retriever_type = type(pipeline.retriever).__name__
    print(f"  Retriever      : {retriever_type}")
    print(f"  Generator      : {type(pipeline.generator.chain).__name__ if pipeline.generator.chain else 'TemplateChain'}")

    print("\n  Running pipeline.answer() with conditioned query...")
    kwargs = cq.to_pipeline_kwargs()
    result = pipeline.answer(**kwargs)

    return result, provider


def display_result(result, provider):
    _hr("STEP 4 — Results")

    print(f"  LLM provider   : {result.get('llm_provider', provider)}")
    print(f"  Risk level     : {result['risk_level']}")
    print(f"  Prediction     : {result['prediction']:.1f} mg/dL")
    print(f"  Query used     :")
    print(f"  >>> {result['query']}")

    print("\n  --- Advisory explanation ---")
    explanation = result.get("explanation", "N/A")
    # Print in chunks of 90 chars for readability
    for i in range(0, len(explanation), 90):
        print(f"  {explanation[i:i+90]}")

    print("\n  --- Doctor advisory summary ---")
    advisory = result.get("advisory", {})
    print(f"  Risk    : {advisory.get('risk_level', 'N/A')}")
    print(f"  Summary : {advisory.get('summary', 'N/A')}")
    factors = advisory.get("key_factors", [])
    if factors:
        print(f"  Factors : {', '.join(factors)}")
    actions = advisory.get("actions", [])
    for i, action in enumerate(actions, 1):
        print(f"  Action {i}: {action}")

    print("\n  --- Retrieved documents ---")
    docs = result.get("retrieved_docs", [])
    print(f"  Total retrieved: {len(docs)}")
    for doc in docs:
        print(f"  [{doc['rank']}] source={doc['source']} | similarity={doc['similarity']:.3f}")
        preview = doc['text'][:80].replace('\n', ' ')
        print(f"      {preview}...")

    print("\n  --- Citations ---")
    citations = result.get("citations", [])
    print(f"  Total citations: {len(citations)}")
    for cite in citations[:3]:
        print(f"  - {cite.get('title', cite.get('source', 'N/A'))} ({cite.get('year', 'N/A')})")

    print(f"\n  ✓ E2E test complete — provider={result.get('llm_provider', provider)}")


def run_second_scenario():
    _hr("BONUS — Second scenario: hypoglycemia falling")
    from src.rag.conditioned_query import build_conditioned_query
    from src.rag.pipeline import RAGPipeline

    cq = build_conditioned_query(
        patient_id="ohio_596",
        current_glucose=75.0,
        predicted_glucose=58.0,
        feature_row={"insulin": 2.5, "carbs": 0.0, "activity": 45, "stress": 3},
    )

    print(f"  Conditioned query: {repr(cq)}")
    print(f"  Primary query snippet:")
    print(f"  >>> {cq.primary_query[:150]}...")

    provider = "gemini" if os.getenv("GOOGLE_API_KEY") else "template"
    pipeline = RAGPipeline(llm_provider=provider, embed_provider="sentence-transformers")
    pipeline.build()

    result = pipeline.answer(**cq.to_pipeline_kwargs())
    print(f"\n  Risk: {result['risk_level']}")
    print(f"  Advisory: {result['explanation'][:200]}...")
    print(f"  ✓ Hypoglycemia scenario OK")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RAG E2E Test — Prediction-Conditioned RAG with Gemini")
    print("="*60)

    try:
        state = test_patient_state_module()
        cq = test_conditioned_query(state)
        result, provider = test_pipeline_e2e(cq)
        display_result(result, provider)
        run_second_scenario()

        _hr("ALL TESTS PASSED")
        print("  ✓ PatientState module")
        print("  ✓ PredictionConditionedQueryBuilder")
        print("  ✓ RAGPipeline end-to-end")
        print("  ✓ Hypoglycemia scenario")
        print()

    except Exception as exc:
        import traceback
        print(f"\n  ✗ Test failed: {exc}")
        traceback.print_exc()
        sys.exit(1)
