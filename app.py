"""
Streamlit UI for the tax-law translator. Loads index.json (built by
ingest.py), retrieves the most relevant chunks for the user's question,
and asks the LLM to answer in plain language, grounded only in those
chunks.

Run with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY from a .env file in this folder, if present

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"     # cheap, fast, good enough for this task
TOP_K = 4                      # how many chunks to retrieve per question
INDEX_PATH = "index.json"
MAX_HISTORY_TURNS = 3          # how many previous Q&A exchanges to feed back in
MAX_TOKENS = 600

SYSTEM_PROMPT = """You are a tax-law translator. You explain Singapore income tax
rules in plain, simple English for people with no legal or accounting background.

Rules you must follow:
1. Only use the CONTEXT provided below to answer. Do not use outside knowledge
   of tax law, and do not guess at numbers, rates, or deadlines that aren't in
   the context.
2. If the context doesn't contain enough information to answer confidently,
   say so clearly and suggest what the person should check on iras.gov.sg or
   with a tax professional instead of guessing.
3. Write short sentences. Avoid legal jargon; if you must use a legal term,
   explain it in brackets the first time.
4. Whenever the context supports it, include one short worked example that
   mirrors the person's situation as closely as possible — using realistic
   but illustrative numbers/details from the context (e.g. "For example, if
   you supported your father, who lives with you and earns $2,000 a year,
   you could..."). Do not invent numbers that aren't in the context; if the
   context has no numeric example to draw from, say so instead of making one
   up.
5. You may refer back to earlier questions in this conversation if the
   person is following up on something they already asked, but every factual
   claim must still be grounded in the CONTEXT given for the current
   question, not in what you said earlier.
6. Always end with: "This is a simplified explanation, not tax advice."
"""


@st.cache_resource
def get_client():
    return OpenAI()  # reads OPENAI_API_KEY from environment


@st.cache_data
def load_index():
    if not os.path.exists(INDEX_PATH):
        return None
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array([d["embedding"] for d in data], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return data, embeddings


def retrieve(question, data, embeddings, client, top_k=TOP_K):
    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    q_emb = np.array(q_emb, dtype=np.float32)
    q_emb /= np.linalg.norm(q_emb)

    scores = embeddings @ q_emb  # cosine similarity since both are normalized
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [data[i] for i in top_idx], [float(scores[i]) for i in top_idx]


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


def main():
    st.set_page_config(page_title="Tax Law Translator", page_icon="📄")
    st.title("Singapore Individual Income Tax — Jargon to Plain Language Translator")
    st.caption("Ask a question about income tax and get a plain-English explanation.")

    index = load_index()
    if index is None:
        st.error(
            "No index.json found. Add source documents to the data/ folder and "
            "run `python ingest.py` first."
        )
        return
    data, embeddings = index

    client = get_client()

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. What tax relief can I claim for my parents?")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Looking up relevant tax rules..."):
                chunks, scores = retrieve(question, data, embeddings, client)
                prompt = build_prompt(question, chunks)
                prior_turns = st.session_state.history[:-1][-(MAX_HISTORY_TURNS * 2):]

                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(prior_turns)
                messages.append({"role": "user", "content": prompt})

                response = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=MAX_TOKENS,
                )
                answer = response.choices[0].message.content

            st.markdown(answer.replace("$", "\\$"))


if __name__ == "__main__":
    main()
