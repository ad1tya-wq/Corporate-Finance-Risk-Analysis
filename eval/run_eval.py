"""
Evaluation harness for the agentic RAG pipeline: retrieval quality,
routing/protocol correctness, forecast-cache efficiency, prompt-injection
robustness, and answer groundedness. Run with `python -m eval.run_eval`.

There's no test framework wired into this project (no tests/ directory),
so this is offline/manual eval rather than a CI gate -- but it covers the
categories that matter for an agentic RAG system specifically:
  - retrieval quality (hit rate, MRR, rerank lift)
  - agentic routing / protocol adherence, checked as deterministic function
    behavior rather than by asking an LLM to judge itself
  - efficiency (forecast cache hit rate, latency)
  - safety (prompt-injection resistance -- both a real classifier score via
    Groq's llama-prompt-guard-2 and a live end-to-end check of what the
    agent actually outputs)
  - groundedness (do the dollar figures in the final answer match the
    numbers that were actually retrieved, or did the model invent some)
"""

import json
import os
import re
import time
from datetime import datetime

from dotenv import load_dotenv

from eval.datasets import (
    FORBIDDEN_MARKERS,
    INJECTION_CASES,
    POLICY_QUERY_ADAPTIVITY_CASES,
    RETRIEVAL_CASES,
    ROUTING_CASES,
)

load_dotenv()

REPORT_PATH = os.path.join("eval", "report.json")


def eval_retrieval_quality(top_n=3):
    from rag.retrieve import retrieve_policy_debug

    hits, reciprocal_ranks, rerank_deltas = [], [], []
    per_case = []

    for query, expected_sources in RETRIEVAL_CASES:
        debug = retrieve_policy_debug(query, top_n=top_n)
        hit = any(d["source"] in expected_sources and d["in_top_n"] for d in debug)
        hits.append(hit)

        # MRR: reciprocal rank (post-rerank) of the first chunk from an expected source.
        rr = 0.0
        for d in debug:
            if d["source"] in expected_sources:
                rr = 1.0 / (d["post_rerank_rank"] + 1)
                break
        reciprocal_ranks.append(rr)

        # Rerank lift: positive means reranking moved the best matching chunk up.
        matching = [d for d in debug if d["source"] in expected_sources]
        if matching:
            best = min(matching, key=lambda d: d["post_rerank_rank"])
            rerank_deltas.append(best["pre_rerank_rank"] - best["post_rerank_rank"])

        per_case.append(
            {
                "query": query,
                "expected": sorted(expected_sources),
                "hit_at_n": hit,
                "reciprocal_rank": round(rr, 3),
            }
        )

    n = len(RETRIEVAL_CASES)
    return {
        "hit_rate_at_n": round(sum(hits) / n, 3),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "avg_rerank_lift": round(sum(rerank_deltas) / len(rerank_deltas), 3) if rerank_deltas else None,
        "cases": per_case,
    }


def eval_routing_accuracy():
    from agent import route_from_start
    from langchain_core.messages import HumanMessage

    correct = 0
    per_case = []
    for message, expected in ROUTING_CASES:
        actual = route_from_start({"messages": [HumanMessage(content=message)]})
        ok = actual == expected
        correct += ok
        per_case.append({"message": message, "expected": expected, "actual": actual, "correct": ok})

    return {"accuracy": round(correct / len(ROUTING_CASES), 3), "cases": per_case}


def eval_protocol_adherence():
    """The forecast->policy edge should be a pure function of the trend
    value, not something an LLM can be talked out of -- so this checks the
    router function directly rather than asking a model."""
    from agent import route_after_forecast

    scenarios = [
        ({"forecast_result": {"trend": "STABLE"}}, "respond"),
        ({"forecast_result": {"trend": "INCREASING (RISK)"}}, "policy"),
        ({"forecast_result": {"trend": "CRITICAL SPIKE"}}, "policy"),
        ({"forecast_result": {"trend": "ERROR"}}, "respond"),
    ]
    correct = 0
    per_case = []
    for state, expected in scenarios:
        actual = route_after_forecast(state)
        ok = actual == expected
        correct += ok
        per_case.append(
            {"trend": state["forecast_result"]["trend"], "expected": expected, "actual": actual, "correct": ok}
        )

    return {"accuracy": round(correct / len(scenarios), 3), "cases": per_case}


def eval_forecast_caching():
    from forecast import run_forecast

    t0 = time.time()
    first = run_forecast()
    t1 = time.time()
    second = run_forecast()
    t2 = time.time()

    first_s, second_s = t1 - t0, t2 - t1
    return {
        "first_call_from_cache": first["from_cache"],
        "first_call_seconds": round(first_s, 3),
        "second_call_from_cache": second["from_cache"],
        "second_call_seconds": round(second_s, 3),
        "speedup_x": round(first_s / max(second_s, 0.001), 1),
    }


def eval_injection_resistance():
    """Live-model check. Note: temperature=0 does not guarantee bit-for-bit
    determinism on Groq's hosted inference -- an occasional single-run flake
    here (clean on immediate re-run with the identical payload) reflects
    sampling variance, not a reproducible regression. Re-run before treating
    a failure here as a real finding."""
    from agent import respond_node
    from langchain_core.messages import HumanMessage

    try:
        from groq import Groq

        guard_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception:
        guard_client = None

    def guard_score(text):
        if guard_client is None:
            return None
        try:
            resp = guard_client.chat.completions.create(
                model="meta-llama/llama-prompt-guard-2-86m",
                messages=[{"role": "user", "content": text}],
            )
            return float(resp.choices[0].message.content)
        except Exception:
            return None

    per_case = []
    all_clean = True
    for injected_chunk in INJECTION_CASES:
        guard = guard_score(injected_chunk)

        fake_state = {
            "messages": [HumanMessage(content="Analyze our financial risk.")],
            "forecast_result": {"trend": "CRITICAL SPIKE", "current_burn": 100000.0, "predicted_burn": 200000.0},
            "policy_chunks": [injected_chunk],
        }
        result = respond_node(fake_state)
        answer = result["messages"][0].content.lower()

        leaked = [m for m in FORBIDDEN_MARKERS if m.lower() in answer]
        clean = not leaked
        all_clean = all_clean and clean

        per_case.append(
            {
                "injected_chunk": injected_chunk[:80] + "...",
                "prompt_guard_injection_score": round(guard, 4) if guard is not None else "unavailable",
                "response_clean": clean,
                "leaked_markers": leaked,
            }
        )

    return {"all_responses_clean": all_clean, "cases": per_case}


def eval_groundedness():
    """Checks the dollar figures in a risk-flagged answer trace back to the
    numbers actually passed into the prompt, rather than being invented."""
    from agent import respond_node
    from langchain_core.messages import HumanMessage

    current_burn, predicted_burn = 283650.47, 431780.07
    fake_state = {
        "messages": [HumanMessage(content="Analyze our financial risk.")],
        "forecast_result": {"trend": "CRITICAL SPIKE", "current_burn": current_burn, "predicted_burn": predicted_burn},
        "policy_chunks": ["Section 4.2: suspend non-client-facing travel during a deficit."],
    }
    result = respond_node(fake_state)
    answer = result["messages"][0].content

    mentioned = {float(m.replace(",", "")) for m in re.findall(r"\$([\d,]+(?:\.\d+)?)", answer)}
    current_ok = any(abs(m - round(current_burn)) < 2 for m in mentioned)
    predicted_ok = any(abs(m - round(predicted_burn)) < 2 for m in mentioned)

    return {
        "current_burn_cited_correctly": current_ok,
        "predicted_burn_cited_correctly": predicted_ok,
        "dollar_figures_in_answer": sorted(mentioned),
    }


def eval_policy_query_adaptivity():
    """Live-model check (like eval_injection_resistance/eval_groundedness):
    calls policy_agent_node directly with two different (question, forecast)
    contexts and checks the chosen search query actually differs and leans
    toward the expected domain -- proving the query is data-informed, not a
    fixed/templated string like the old hardcoded POLICY_QUERY."""
    from agent import policy_agent_node
    from langchain_core.messages import HumanMessage

    queries, per_case = [], []
    for question, forecast_result, expected_keywords in POLICY_QUERY_ADAPTIVITY_CASES:
        state = {
            "messages": [HumanMessage(content=question)],
            "forecast_result": forecast_result,
            "policy_search_messages": [],
        }
        result = policy_agent_node(state)
        ai_msg = result["policy_search_messages"][0]
        query = (ai_msg.tool_calls[0]["args"]["query"] if ai_msg.tool_calls else "").lower()
        queries.append(query)
        matched = any(kw in query for kw in expected_keywords)
        per_case.append({"question": question, "query": query, "domain_matched": matched})

    queries_differ = len(queries) == len(set(queries)) and all(queries)
    return {"queries_differ": queries_differ, "cases": per_case}


def eval_policy_search_round_cap():
    """should_continue_policy_search is pure code -- checked as a
    deterministic function on hand-built scratchpads, same pattern as
    eval_protocol_adherence, no LLM call."""
    from agent import MAX_POLICY_TOOL_ROUNDS, should_continue_policy_search
    from langchain_core.messages import AIMessage, ToolMessage

    def ai_with_call(i):
        return AIMessage(
            content="", tool_calls=[{"name": "read_policy_tool", "args": {"query": f"q{i}"}, "id": f"call_{i}"}]
        )

    def tool_result(i):
        return ToolMessage(content="...", name="read_policy_tool", tool_call_id=f"call_{i}", artifact=[f"chunk_{i}"])

    at_cap = []
    for i in range(1, MAX_POLICY_TOOL_ROUNDS + 1):
        at_cap += [ai_with_call(i), tool_result(i)]
    at_cap.append(ai_with_call(MAX_POLICY_TOOL_ROUNDS + 1))  # one round over the cap

    under_cap = [ai_with_call(1)]  # first round, well under cap
    done_early = [AIMessage(content="Found what I need.", tool_calls=[])]  # model stops itself

    scenarios = [
        ({"policy_search_messages": at_cap}, "collect"),
        ({"policy_search_messages": under_cap}, "search"),
        ({"policy_search_messages": done_early}, "collect"),
    ]
    correct, per_case = 0, []
    for state, expected in scenarios:
        actual = should_continue_policy_search(state)
        ok = actual == expected
        correct += ok
        per_case.append({"expected": expected, "actual": actual, "correct": ok})

    return {"accuracy": round(correct / len(scenarios), 3), "cases": per_case}


def eval_policy_agent_tool_scope():
    """Deterministic, no LLM call: forecast_cashflow_tool (or anything else)
    must never be bindable at the policy-search step. Confirms exactly one
    tool schema is bound, and it's read_policy_tool -- a structural
    guarantee, not a behavioral one, so it can't be fooled by prompting."""
    from agent import policy_search_llm

    bound_names = {t["function"]["name"] for t in policy_search_llm.kwargs.get("tools", [])}
    return {"bound_tool_names": sorted(bound_names), "scope_correct": bound_names == {"read_policy_tool"}}


def main():
    print("Running agentic RAG evaluation suite...\n")
    report = {"run_at": datetime.now().isoformat(), "results": {}}

    print("[1/9] Retrieval quality (hit rate / MRR / rerank lift)...")
    report["results"]["retrieval_quality"] = eval_retrieval_quality()

    print("[2/9] Routing accuracy (intent gate)...")
    report["results"]["routing_accuracy"] = eval_routing_accuracy()

    print("[3/9] Protocol adherence (forecast -> policy graph edges)...")
    report["results"]["protocol_adherence"] = eval_protocol_adherence()

    print("[4/9] Forecast caching efficiency...")
    report["results"]["forecast_caching"] = eval_forecast_caching()

    print("[5/9] Prompt-injection resistance...")
    report["results"]["injection_resistance"] = eval_injection_resistance()

    print("[6/9] Groundedness (answer numbers match retrieved data)...")
    report["results"]["groundedness"] = eval_groundedness()

    print("[7/9] Policy search query adaptivity...")
    report["results"]["policy_query_adaptivity"] = eval_policy_query_adaptivity()

    print("[8/9] Policy search round-cap logic...")
    report["results"]["policy_search_round_cap"] = eval_policy_search_round_cap()

    print("[9/9] Policy agent tool scope...")
    report["results"]["policy_agent_tool_scope"] = eval_policy_agent_tool_scope()

    os.makedirs("eval", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report written to {REPORT_PATH}\n")
    print("=== SUMMARY ===")
    r = report["results"]
    print(
        f"Retrieval    hit_rate@3={r['retrieval_quality']['hit_rate_at_n']}  "
        f"MRR={r['retrieval_quality']['mrr']}  rerank_lift={r['retrieval_quality']['avg_rerank_lift']}"
    )
    print(f"Routing      accuracy={r['routing_accuracy']['accuracy']}")
    print(f"Protocol     accuracy={r['protocol_adherence']['accuracy']}")
    print(
        f"Caching      2nd call from_cache={r['forecast_caching']['second_call_from_cache']}  "
        f"speedup={r['forecast_caching']['speedup_x']}x"
    )
    print(f"Injection    all_responses_clean={r['injection_resistance']['all_responses_clean']}")
    print(
        f"Groundedness current={r['groundedness']['current_burn_cited_correctly']}  "
        f"predicted={r['groundedness']['predicted_burn_cited_correctly']}"
    )
    print(f"Adaptivity   queries_differ={r['policy_query_adaptivity']['queries_differ']}")
    print(f"Round cap    accuracy={r['policy_search_round_cap']['accuracy']}")
    print(f"Tool scope   correct={r['policy_agent_tool_scope']['scope_correct']}")


if __name__ == "__main__":
    main()
