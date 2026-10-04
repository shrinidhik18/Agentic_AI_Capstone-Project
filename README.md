# AI-Based College Academic Assistant

🚀 **Live Demo:** [https://agenticaicapstone-project-lxedmhb457pljbp2m4mbtx.streamlit.app/](https://agenticaicapstone-project-lxedmhb457pljbp2m4mbtx.streamlit.app/)

An AI-powered academic assistant for college students that answers academic questions by retrieving information from official college documents (syllabus, academic regulations, examination guidelines, internship guidelines, student FAQs) and builds personalized study plans using LangGraph stateful orchestration and FAISS RAG vector retrieval.

---

## 🌟 Key Features

1. **Document Ingestion & Chunking Pipeline**:
   - Ingests PDFs, TXT, and Markdown documents from `data/sample_documents/` or via Web UI uploads.
   - Automatically tags chunks with metadata classification (`syllabus`, `regulation`, `examination`, `internship`, `faq`).
   - Persists vector embeddings in FAISS local vector database.

2. **Grounded RAG Q&A with Review Node**:
   - Retrieves top-$k$ relevant chunks using relevance thresholding.
   - Groundedness Review Node evaluates draft answers against source context before returning to student.
   - Explicit "Information Not Available in Official Documents" fallback prevents hallucinations.

3. **LangGraph Stateful Orchestration**:
   - `classify_intent` -> routes query to `document_qa`, `create_study_plan`, `modify_study_plan`, or `unrecognized`.
   - `retrieve_node` -> vector search & relevance filtering.
   - `generate_response_node` -> context-grounded response generation.
   - `review_response_node` -> groundedness verification and claim filtering.
   - `study_planner_node` -> calendar date-math & schedule modification.

4. **Calendar Date-Math & Study Planning Tool**:
   - Uses real date arithmetic (`CalendarDateTool`) to compute days remaining, weekday/weekend hour allocations, and day-by-day timetable.
   - Supports natural language modifications (e.g., *"Move all Data Structures sessions to the morning"*).

5. **Head-to-Head RAG vs. Plain LLM Comparison**:
   - Evaluates grounded accuracy, latency, and hallucination contrast between RAG and unassisted LLM calls.
   - Includes full 5-category benchmark suite corresponding to Section 9 of the PRD.

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Copy `.env.example` to `.env` and add your OpenAI or Google Gemini API Key:
```env
OPENAI_API_KEY=your_openai_api_key_here
# OR
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(Note: If no API keys are provided, the system automatically uses the embedded standalone local fallback engine for zero-dependency local runs!)*

### 3. Launch the Web Application
Run the Streamlit application:
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
├── app.py                      # Main Streamlit Web Application Interface
├── backend/
│   ├── __init__.py
│   ├── llm_config.py           # LLM provider initialization & local fallback engine
│   ├── ingestion.py            # Document parsing, metadata classification & chunking
│   ├── vectorstore.py          # FAISS vector database manager & threshold search
│   ├── tools.py                # CalendarDateTool & CalculatorTool for academic math
│   ├── workflow.py             # LangGraph state machine workflow nodes & routing
│   └── evaluator.py            # RAG vs Plain LLM evaluation & 5-category test suite
├── data/
│   └── sample_documents/       # Pre-loaded sample college policies & syllabi
│       ├── academic_regulations.md
│       ├── examination_guidelines.md
│       ├── internship_guidelines.md
│       ├── syllabus_cs.md
│       └── student_faqs.md
├── vector_db/                  # Local persisted FAISS index files
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🎯 Verification & Benchmark Test Suite
You can execute the automated PRD test suite directly:
```bash
python scratch_test.py
```
This tests all 5 PRD categories:
1. **Direct Questions** (Single document lookup)
2. **Follow-up Questions** (Conversational context)
3. **RAG vs Plain LLM Comparison**
4. **Unknown / Out-of-Scope Questions** (Explicit fallback verification)
5. **Multi-Step Study Planning** (Creation & Modification)
