# Scholera AI Tutor

Course-grounded **FastAPI** backend prototype for **teaching tools**: a **hybrid RAG AI Tutor**, a **lightweight Quiz Generator**, and a **local dashboard**—all backed by the **same persisted per-course index** (chunks + embeddings + metadata). Ingest **PDF** and **PPTX** once; answer and quiz from **retrieved excerpts** with **citations**, **abstention** when evidence does not support the question, and **retrieval debug** for transparency.

**Generation:** when **`GOOGLE_API_KEY`** is set, answers and quizzes use **Gemini** via **Google ADK**; otherwise the tutor uses a **deterministic local fallback** (quiz requires the API key). The stack always runs retrieval locally; only the **wording** step is cloud-backed when configured.

---

## 1. Project overview

Scholera builds a **reusable knowledge base** per `course_id` under `data/courses/{course_id}/`:

- **Parse** PDFs page-by-page (**PyMuPDF**) and PPTX slide-by-slide (**python-pptx**).
- **Chunk** with page/slide metadata preserved (`app/ingestion/chunker.py`).
- **Embed** each chunk with **sentence-transformers** (default **`all-MiniLM-L6-v2`**, L2-normalized).
- **Retrieve** with **hybrid dense + BM25** fusion (`app/retrieval/hybrid.py`).
- **Tutor** (`POST .../ask`): hybrid search → optional **concept-coverage** and **keyword-gap** abstention → **Gemini** or **template/sentence fallback** → citations aligned with evidence.
- **Quiz** (`POST .../quiz`): same retrieval pipeline → diversified or topic-focused context → **Gemini** emits JSON → **citation whitelist** so only retrieved sources appear.

A **single-page dashboard** at **`GET /finalUI`** drives ingest, stats, ask, and quiz against the running API.

---

## 2. Features

- **AI Tutor** — grounded Q&A with hybrid retrieval, synthesis context selection, and citations  
- **Quiz Generator** — short quizzes with questions, answer keys, and validated citations (lightweight, not a full LMS)  
- **Persistent per-course index** — `chunks.json`, `embeddings.npy`, ingestion summary on disk; reloads on startup  
- **Hybrid retrieval** — cosine dense + BM25, min–max normalized per query, weighted fusion, dedupe, deterministic ordering  
- **Citations** — `source_file`, `unit_type` (`page` / `slide`), `unit_number` on supported answers  
- **Abstention** — concept-coverage gate, keyword-gap gate, weak-evidence caveats; abstentions list **reviewed sources** without implying evidentiary support  
- **Local dashboard** — `/finalUI` (static HTML/CSS/JS, no build step)  
- **Ingestion summary** — formula/sparse heuristics and warnings  

---

## 3. Architecture overview

```text
┌─────────────┐   multipart    ┌──────────────────┐
│   Client    │ ─────────────► │ FastAPI routers  │
└─────────────┘                └────────┬─────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │ Ingestion pipeline (load/chunk/detect) │
                    └───────────────────┬───────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          ▼                             ▼                             ▼
   ┌─────────────┐              ┌─────────────┐               ┌─────────────┐
   │ Embeddings  │              │ Vector mat  │               │ BM25 index  │
   │ (local ST)  │              │ (cosine)    │               │ (rank_bm25) │
   └──────┬──────┘              └──────┬──────┘               └──────┬──────┘
         │                            │                             │
         └────────────────────────────┼─────────────────────────────┘
                                      ▼
                            ┌──────────────────┐
                            │ Hybrid fusion    │
                            └────────┬─────────┘
                                     ▼
              ┌──────────────────────┴──────────────────────┐
              ▼                                              ▼
     ┌─────────────────┐                          ┌─────────────────┐
     │ AI Tutor        │                          │ Quiz generator  │
     │ (abstain →      │                          │ (JSON + cite    │
     │  Gemini /       │                          │  validation)    │
     │  fallback)      │                          └─────────────────┘
     └─────────────────┘
```

Persistence: **chunk JSON**, **NumPy embedding matrix**, **last ingestion JSON**. BM25 is built **in memory** at query time (fine for assignment-scale corpora).

---

## 4. How ingestion works

1. **Upload** to `POST /courses/{course_id}/ingest` (multipart `files`).  
2. **Type detection** by extension (`.pdf`, `.pptx`); unsupported types skipped with warnings.  
3. **`lecture_id`** from filename stem (`slugify_filename_stem`).  
4. **Parsing** — PDF: PyMuPDF per page; PPTX: python-pptx per slide.  
5. **Heuristics** per unit → chunk flags: `has_formula`, `is_sparse_or_image_heavy`.  
6. **Chunking** — short units → one chunk; long units split on paragraphs then overlapping windows (~300–700 words, ~50–100 overlap).  
7. **Embeddings** — sentence-transformers, L2-normalized rows.  
8. **Persistence** — `data/courses/{course_id}/`.  
9. **Re-ingest** — same `lecture_id` replaces prior chunks for that lecture.

---

## 5. How hybrid retrieval works

1. **Dense** — cosine similarity vs chunk embeddings (`app/retrieval/vector_index.py`).  
2. **Lexical** — BM25 on tokenized chunk texts (`app/retrieval/bm25_index.py`).  
3. **Normalize** — independent min–max to \([0,1]\) per query per channel.  
4. **Fuse** — \(s_{\text{hybrid}} = w_d \hat{s}_d + w_b \hat{s}_b\); defaults **`HYBRID_WEIGHT_DENSE=0.65`**, **`HYBRID_WEIGHT_BM25=0.35`** (renormalized if needed).  
5. **No lexical tokens** → dense-only.  
6. **Dedup** by `chunk_id`; **sort** hybrid desc, dense desc, BM25 desc, `chunk_id` asc.

**Cross-lecture questions** (heuristic markers in `query_plan.py`): larger retrieval pool, then **`diversify_by_source_file`** so final `top_k` often spans multiple files.

---

## 6. AI Tutor generation

`POST /courses/{course_id}/ask`:

1. Load chunks + embedding matrix; run **hybrid retrieval** (with diversification when the question looks cross-lecture).  
2. **Concept coverage** (`app/retrieval/concept_coverage.py`) for definitional / explanatory / general questions: extracted **core concepts** must appear in top retrieved text (with a small-keyword fallback for long “what are…” prompts); otherwise **abstain**.  
3. **Keyword-gap abstention** (`should_abstain_for_keyword_gap`) as a second line of defense.  
4. **Context selection** (`app/generation/context_select.py`) — re-ranks chunks for synthesis; `debug.retrieved_chunks` stays the full retrieval picture; **citations** follow the evidence used (capped).  
5. **Weak evidence** — if top hybrid score **&lt; 0.12**, a caveat is prepended and answers stay conservative.  
6. **Generation** (`app/generation/tutor.py`):  
   - **With `GOOGLE_API_KEY`:** **Google ADK** + **Gemini** (`LLM_MODEL`, default `gemini-2.5-flash`) with a grounded system instruction; optional **Google Search** tool for the tutor path.  
   - **Without key or on API failure:** **deterministic fallback** — question-type routing (`app/generation/question_types.py`) and templates (`app/generation/fallback_templates.py`), plus sentence-level ranking for general questions.  
7. **Abstention responses** — `citations` are **empty**; the answer includes **reviewed sources** (closest retrieval, not framed as supporting evidence).

---

## 7. Quiz Generator (implemented)

`POST /courses/{course_id}/quiz` with JSON body: `num_questions`, optional `topic`, `difficulty` (`easy` | `medium` | `hard`), optional `top_k`.

- **Retrieval** — same `CourseRepository` + `hybrid_search`. With **`topic`**, embeds the topic string and retrieves top‑k. **Without topic**, a broad query plus **`diversify_by_source_file`** encourages coverage across files. Very weak topic matches may return an empty quiz with an explanatory `message` before calling the model.  
- **Generation** — a dedicated **ADK agent** (no web search) prompts Gemini to output **JSON only**: questions, answer keys, and citations.  
- **Validation** — citations must match **exact retrieved** `(source_file, page|slide)`; numeric `page` / `slide` values tolerate JSON **int/float** coercion. Unsupported or hallucinated citations are dropped; **questions are filtered** so unsupported or uncited items are **not returned** when validation fails.  
- **Intent** — **lightweight** professor aid: not a calibrated assessment engine, not proctored exam software. Difficulty is **prompt-level guidance**, not psychometrically tuned. Quality depends on **retrieval and excerpt coverage**; repetitive stems can occur on thin corpora.

**Gemini quota:** free-tier **rate limits / daily quotas** can return HTTP **429** during heavy testing; the API surfaces that in `message`. Use **`LLM_MODEL`** to switch models, wait for reset, or enable billing per [Google’s limits](https://ai.google.dev/gemini-api/docs/rate-limits).

---

## 8. API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Short JSON pointer (`/docs`, `/health`, `/finalUI`). |
| `GET` | `/health` | Liveness-style ping. |
| `GET` | `/courses` | List `course_id` values with an on-disk index. |
| `GET` | `/finalUI` | **Local dashboard** (ingest, stats, ask, quiz). |
| `POST` | `/courses/{course_id}/ingest` | Multipart PDF/PPTX upload → `IngestionSummary`. |
| `POST` | `/courses/{course_id}/ask` | JSON `AskRequest` → `AskResponse` (answer, citations, debug). |
| `POST` | `/courses/{course_id}/quiz` | JSON `QuizRequest` → `QuizResponse` (quiz items + optional `message`). |
| `GET` | `/courses/{course_id}/stats` | Corpus stats + `recent_ingestion`. |

Interactive docs: **`http://127.0.0.1:8000/docs`**.

### 8.1 Local dashboard (`/finalUI`)

Single-page UI (`app/ui/final_dashboard.html`): **Ingest**, **Course stats**, **Ask a question**, **Quiz generator**. Last selected course id is stored in **`localStorage`** (`scholera_last_course_id`). Empty course list shows a hint on the stats panel.

---

## 9. Environment variables

Copy **`.env.example`** → **`.env`**. Relevant variables:

| Variable | Role |
|----------|------|
| `GOOGLE_API_KEY` | Gemini (ADK) for tutor + quiz; omit for tutor **fallback-only** (quiz needs the key). |
| `LLM_MODEL` | Gemini model id (default `gemini-2.5-flash`). |
| `SCHOLERA_DATA_DIR` | Data root (default `./data`). |
| `EMBEDDING_MODEL` | sentence-transformers name (default `all-MiniLM-L6-v2`). |
| `HYBRID_WEIGHT_DENSE` / `HYBRID_WEIGHT_BM25` | Fusion weights (normalized if they do not sum to 1). |

---

## 10. Local setup and run

**Requirements:** Python **3.11+** (3.13 works in CI).

```bash
cd AI_Scholera
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then set GOOGLE_API_KEY 
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open **`http://127.0.0.1:8000/finalUI`** or **`/docs`**.

**Note:** first run downloads the embedding model (~80–120 MB). Do not suspend the server with **Ctrl+Z** on Unix (port stays bound); use **Ctrl+C** to exit.

---

## 11. Example `curl` flows

**Health**

```bash
curl -s http://127.0.0.1:8000/health
```

**Ingest**

```bash
curl -s -X POST "http://127.0.0.1:8000/courses/ml101/ingest" \
  -F "files=@./lecture1.pdf"
```

**Ask**

```bash
curl -s -X POST "http://127.0.0.1:8000/courses/ml101/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main idea in lecture 1?","top_k":8}'
```

**Quiz**

```bash
curl -s -X POST "http://127.0.0.1:8000/courses/ml101/quiz" \
  -H "Content-Type: application/json" \
  -d '{"num_questions":5,"topic":"clustering","difficulty":"medium","top_k":12}'
```

**Stats / list courses**

```bash
curl -s "http://127.0.0.1:8000/courses/ml101/stats"
curl -s "http://127.0.0.1:8000/courses"
```

---

## 12. Honest limitations

**Extraction and corpus quality**

- Answers are only as good as **parsed text**. Diagram-heavy, image-first, or visually encoded slides often yield **thin text**; `is_sparse_or_image_heavy` flags some cases but does not recover meaning from pixels.  
- **Scanned PDFs** without OCR produce little text and weak retrieval.  
- **Formula-heavy** or noisy math can hurt both BM25 and general-purpose embeddings; `has_formula` is a signal, not a fix.  
- PDF extraction order, hyphenation, and PPTX shape trees can **garble** content before indexing.

**Abstention and grounding**

- Abstention is **real** but **heuristic** (concept phrases, keyword overlap, thresholds). It reduces “wrong lecture” answers but cannot guarantee perfection.  
- **Broadly related** material can still appear at the top of retrieval before abstention fires; “reviewed sources” vs “supporting citations” is clearest on **abstain** paths; the live UI is still simpler than a full audit UI.

**Question types**

- **Abbreviations / “full form of X”** and highly **compositional** synthesis questions are less stable than straightforward factual asks.  
- Narrow **definitions** can misfire when the corpus uses different wording than the question, even if a human would connect them.

**Fallback templates**

- When Gemini is unavailable or fails, **template + extractive** behavior remains; wording and structure can **vary** by question type. More time could unify tone and reduce template “feel.”

**Quiz**

- **Lightweight** by design: occasional **repetition**, **approximate** difficulty (prompt-guided, not calibrated), quality **tied to retrieval** and excerpt length. Not a replacement for item banks or psychometrics.

**API and provider**

- **Gemini free tier** quotas / **429** can interrupt demos; generation quality and uptime depend on **Google’s API**.  
- **Local fallback** helps the tutor run offline; it is **less polished** than API-backed answers. **Quiz** does not have a non-Gemini generator path.

**Product scope**

- **No auth / tenant isolation** — take-home scope.  
- **No background jobs** — large uploads block the request; fine for demos, not batch farms.

**Non-goals (not claimed)**

- No chart CV, no slide vision encoders, no LaTeX symbolic engine — see extraction limits above.

---

## 13. What I would improve next

- **Abbreviation and acronym expansion** — targeted retrieval or light entity linking before abstention.  
- **Synthesis routing** — stronger handling for multi-clause “compare A vs B under constraint C” questions.  
- **Abstention / concept coverage** — tighter coupling between retrieval score and concept gates; optional second-stage reranker.  
- **Quiz** — MMR-style excerpt diversity, de-duplication of stems, optional `lecture_id` / date filters in the request.  
- **Image- and formula-heavy units** — optional OCR path or vision-language encoding behind a flag.  
- **Answer consistency** — fewer template branches when the API is off; single “house style” post-processor.  
- **Eval harness** — fixed Q/A sets for regression on retrieval and abstention.

---

## 14. Demo checklist

- [ ] Fresh venv: `pip install -r requirements.txt`.  
- [ ] `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` starts; open **`/finalUI`**.  
- [ ] Ingest **≥2** files (mix PDF + PPTX if possible) under one `course_id`.  
- [ ] **`/stats`** — chunks, lectures, `indexed_files` look sensible.  
- [ ] **Ask** — one **narrow factual** question and one **cross-lecture** question; expand retrieval debug.  
- [ ] **Ask** — one **off-corpus** concept (e.g. unrelated ML buzzword) and confirm **abstention** + empty supporting citations + reviewed sources.  
- [ ] **Quiz** — generate with and without `topic`; confirm citations only reference real files/pages from your corpus.  
- [ ] **Optional** — repeat **ask** with `GOOGLE_API_KEY` unset to see **fallback** tutor behavior.

---

## Repository layout

```text
app/
  main.py
  config.py
  schemas.py
  api/           # ingest, stats, tutor, quiz, health, and course routes
  ingestion/
  retrieval/     # hybrid, bm25_index, vector_index, embeddings, query_plan, concept_coverage
  generation/    # tutor, quiz, fallback_templates, context_select, question_types
  storage/
  ui/            # final_dashboard.html
  utils/
data/            # local artifacts (gitignored under courses/)
tests/
README.md
requirements.txt
.env.example
pyproject.toml
```

---

## 15. Demo video

- **Demo video:** https://drive.google.com/file/d/1S64md2LGSFlhiRFMKm8YcnXgGM2NdZTM/view?usp=sharing

---

## License

This project scaffold is provided for evaluation / educational use unless you attach your own license.
