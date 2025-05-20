"""
Template to run A/B pairwise evaluations on LLM responses. It loads evaluation results from `Evaluations.ipynb`, prepares comparison pairs, records votes, computes overall performance metrics, and displays results in a Streamlit UI.
"""

import streamlit as st
import pandas as pd
import random
import json
from collections import defaultdict
from itertools import combinations

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-GENERATED DATA
# ─────────────────────────────────────────────────────────────────────────────
EVAL_RESULTS_FILE = "ab_test_data_output.json"

def load_evaluation_results(filepath):
    """Load evaluation JSON."""
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)
        st.sidebar.success(f"Loaded results from {filepath}")
        return data
    except FileNotFoundError:
        st.sidebar.warning(f"{filepath} not found!")
    except json.JSONDecodeError:
        st.sidebar.error(f"JSON decode error in {filepath}")
    except Exception as e:
        st.sidebar.error(f"Unexpected error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PREPARE PAIRS - A/B 
# ─────────────────────────────────────────────────────────────────────────────
def prepare_ab_test_data(results) :
    """Returns [(prompt, model_A, resp_A, model_B, resp_B), …]"""
    prompt_responses = defaultdict(list)

    if isinstance(results, list):
        for row in results:
            prompt = row.get("prompt", "").strip()
            model = row.get("model_name", "").strip()
            resp = row.get("response", "").strip()
            if prompt and model and resp:
                prompt_responses[prompt].append({"model": model, "response": resp})

    pairs = []
    for prompt, resps in prompt_responses.items():
        if len(resps) < 2:
            continue
        for r1, r2 in combinations(resps, 2):
            pairs.append(
                (prompt, r1["model"], r1["response"], r2["model"], r2["response"])
            )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# SCORE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ab_results(votes):   
    model_wins, model_ties, model_losses, model_comparisons = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    head_to_head = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "ties": 0, "losses": 0})
    )

    all_models = set()

    for key, result in votes.items():
        model1, model2 = key[0], key[1]
        all_models.update([model1, model2])
        model_comparisons[model1] += 1
        model_comparisons[model2] += 1

        winner = result.get("winner")
        if winner == model1:
            model_wins[model1] += 1
            model_losses[model2] += 1
            head_to_head[model1][model2]["wins"] += 1
            head_to_head[model2][model1]["losses"] += 1
        elif winner == model2:
            model_wins[model2] += 1
            model_losses[model1] += 1
            head_to_head[model2][model1]["wins"] += 1
            head_to_head[model1][model2]["losses"] += 1
        else:
            model_ties[model1] += 1
            model_ties[model2] += 1
            head_to_head[model1][model2]["ties"] += 1
            head_to_head[model2][model1]["ties"] += 1

    summary_df = pd.DataFrame(
        [
            {
                "Model": m,
                "Wins": model_wins[m],
                "Ties": model_ties[m],
                "Losses": model_losses[m],
                "Total Comparisons": model_comparisons[m],
                "Win Rate (%)": (model_wins[m] / model_comparisons[m]) * 100
                if model_comparisons[m]
                else 0,
            }
            for m in sorted(all_models)
        ]
    ).set_index("Model")
    summary_df.sort_values("Win Rate (%)", ascending=False, inplace=True)

    summary_styler = summary_df.style.format({"Win Rate (%)": "{:.1f}%"}).background_gradient(
        cmap="Greens", subset=["Win Rate (%)"]
    )

    h2h = pd.DataFrame(index=sorted(all_models), columns=sorted(all_models))
    for m1 in h2h.index:
        for m2 in h2h.columns:
            h2h.loc[m1, m2] = (
                "--"
                if m1 == m2
                else "{wins} / {ties} / {losses}".format(**head_to_head[m1][m2])
            )

    return summary_styler, h2h


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("A/B Pairwise LLM Response Evaluation")

all_evaluation_results = load_evaluation_results(EVAL_RESULTS_FILE)

if "comparison_pairs" not in st.session_state:
    st.session_state.comparison_pairs = prepare_ab_test_data(all_evaluation_results)
    random.shuffle(st.session_state.comparison_pairs)
    st.session_state.current_pair_index = 0
    st.session_state.votes = {}
    st.session_state.evaluation_complete = False
    st.session_state.last_assignment = None

if st.session_state.current_pair_index >= len(st.session_state.comparison_pairs):
    st.session_state.evaluation_complete = True

# ── RESULTS VIEW ────────────────────────────────────────────────────────────
if st.session_state.evaluation_complete:
    st.header("Evaluation Complete!")
    st.balloons()

    if not st.session_state.votes:
        st.warning("No votes were recorded.")
    else:
        summary, h2h = calculate_ab_results(st.session_state.votes)
        st.subheader("Overall Model Performance")
        st.dataframe(summary, use_container_width=True)

        st.subheader(" Head‑to‑Head Results (Wins / Ties / Losses)")
        st.dataframe(h2h, use_container_width=True)

        if st.button("🔄 Start New Evaluation"):
            for k in [
                "comparison_pairs",
                "current_pair_index",
                "votes",
                "evaluation_complete",
                "last_assignment",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

# ── VOTING UI ───────────────────────────────────────────────────────────────
else:
    if not st.session_state.comparison_pairs:
        st.error("No comparison pairs could be generated from the results.")
        st.stop()

    total_pairs = len(st.session_state.comparison_pairs)
    idx = st.session_state.current_pair_index
    prompt, m1, r1, m2, r2 = st.session_state.comparison_pairs[idx]

    st.header(f"Pair {idx + 1} of {total_pairs}")
    st.info(f"**Prompt:** {prompt}")

    models = [(m1, r1), (m2, r2)]
    random.shuffle(models)
    model_a, resp_a = models[0]
    model_b, resp_b = models[1]

    st.session_state.last_assignment = {
        "A": {"model": model_a, "response": resp_a},
        "B": {"model": model_b, "response": resp_b},
        "prompt": prompt,
    }

    col1, col2 = st.columns(2)
    col1.markdown("**Response A**")
    col1.markdown(f"> {resp_a}")
    col2.markdown("**Response B**")
    col2.markdown(f"> {resp_b}")

    st.markdown("---")
    st.subheader("Which response is better?")

    def record_vote(choice: str):
        a = st.session_state.last_assignment
        model_a, model_b, prompt = a["A"]["model"], a["B"]["model"], a["prompt"]

        vote_key = tuple(sorted((model_a, model_b))) + (prompt,)
        winner = (
            model_a
            if choice == "A"
            else model_b
            if choice == "B"
            else "tie"
        )

        st.session_state.votes[vote_key] = {"winner": winner}
        st.session_state.current_pair_index += 1
        st.session_state.last_assignment = None
        st.rerun()

    btn_cols = st.columns(3)
    btn_cols[0].button("⬅️ Response A is Better", on_click=record_vote, args=("A",), use_container_width=True, type="primary")
    btn_cols[1].button("Tie / Cannot Decide", on_click=record_vote, args=("Tie",), use_container_width=True)
    btn_cols[2].button("Response B is Better ➡️", on_click=record_vote, args=("B",), use_container_width=True, type="primary")

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
if isinstance(all_evaluation_results, list):
    model_count = len({row["model_name"] for row in all_evaluation_results})
else:
    model_count = len(all_evaluation_results)

st.sidebar.header("Controls & Info")
st.sidebar.write(f"Total Models Found: **{model_count}**")
st.sidebar.write(f"Total Comparison Pairs: **{len(st.session_state.comparison_pairs)}**")
st.sidebar.write(f"Current Pair Index: **{st.session_state.current_pair_index}**")

if (
    not st.session_state.evaluation_complete
    and st.sidebar.button("Skip Current Pair")
):
    st.session_state.current_pair_index += 1
    st.rerun()
