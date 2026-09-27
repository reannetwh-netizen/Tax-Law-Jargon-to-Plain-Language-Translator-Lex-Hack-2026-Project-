# Project Name: Tax Law Jargon to Plain Language Translator 

## Summary
A small Retrieval-Augmented Generation tool that answers Singapore income tax
questions in plain English.

## Video Demo
https://drive.google.com/file/d/1beQKsvY74n-GAyVbRsQXWVk-VUmiZLTX/view?usp=sharing


## Tech Stack 
Language: Python 3.9.6 
AI/LLM: Open AI API (gpt-4o-mini and text-embedding-3-small) + Retrieval-Augmented Generation (RAG)
Data Ingestion: pypdf and tiktoken
Retrieval/Search: NumPy and JSON
Frontend/UI: Streamlit

## 1. Install

cd tax-rag-tool
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt


## 2. Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys and create a key.
2. Set it as an environment variable

Cost note: `text-embedding-3-small` and `gpt-4o-mini` (used in this project) are OpenAI's cheapest capable models — indexing a few dozen PDFs and asking a few hundred questions will typically cost well under $1.

## 3. Add the data

Drop `.txt` or `.pdf` files into the `data/` folder 

## 4. Build the index

Run this once, and again any time you add/change files in `data/`:
```bash
python ingest.py
```
This reads every file in `data/`, splits it into chunks, embeds each chunk, and saves everything to `index.json`.

## 5. Run the app

```bash
streamlit run app.py
```

This opens a browser tab with a chat interface. Ask a question, and the tool
will:
1. Embed your question
2. Retrieve the most relevant chunks from `index.json`
3. Ask the LLM to answer using **only** those chunks
4. Show the answer plus which source documents it drew from

## Important

This is a student/demo project. It is not tax advice and should not be
relied on for real filing decisions — always direct real use back to
iras.gov.sg or a tax professional.

## CITATIONS
PDFs on Individual Income Tax from IRAS offical website: 
https://www.iras.gov.sg/quick-links/e-tax-guides?taxtype=individual-income-tax&industries=&topics=&years=&keyword=income%20tax&sort=datedesc&page=1
https://www.iras.gov.sg/docs/default-source/uploadedfiles/pdf/4a-quick-guide-for-e-filers-2.pdf?sfvrsn=ffaae252_43
https://file.go.gov.sg/efilingguide.pdf

Websites sharing tax advice: 
https://www.stashaway.sg/r/lower-income-tax-singapore
https://taxsummaries.pwc.com/singapore/individual/taxes-on-personal-income
https://www.hawksford.com/insights-and-guides/singapore-expat-income-tax-guide
https://www.expat.hsbc.com/expat-explorer/expat-guides/singapore/tax-in-singapore/


