import time
import json
import logging
from typing import Dict, Any, List
from backend.workflow import WorkflowPipeline
from backend.llm_config import get_llm
from backend.prompt_templates import PLAIN_LLM_TEMPLATE

logger = logging.getLogger("SystemEvaluator")

PRD_BENCHMARK_SUITE = [
    {
        "id": "CAT1-DIRECT",
        "category": "1. Direct Questions (Single Document)",
        "query": "What is the minimum attendance required at NITTE to sit the semester end examination?",
        "expected_doc": "nitte_academic_regulations.md",
        "expected_fact": "85%"
    },
    {
        "id": "CAT2-FOLLOWUP",
        "category": "2. Follow-up Questions (Conversational Context)",
        "query": "What happens if I fall below 75% attendance at NITTE?",
        "context_history": [
            {"role": "user", "content": "What is the minimum attendance required at NITTE to sit the semester end examination?"},
            {"role": "assistant", "content": "Minimum attendance required at NITTE is 85%."}
        ],
        "expected_doc": "nitte_academic_regulations.md",
        "expected_fact": "N grade / debarred / detained from SEE"
    },
    {
        "id": "CAT3-RAG-VS-PLAIN",
        "category": "3. RAG vs Plain LLM Comparison",
        "query": "What is the CIE and SEE weightage for theory courses at NITTE?",
        "expected_doc": "nitte_academic_regulations.md",
        "expected_fact": "CIE 60 marks, SEE 40 marks"
    },
    {
        "id": "CAT4-UNKNOWN",
        "category": "4. Unknown / Out-of-Scope Questions",
        "query": "What is the best restaurant near NMAMIT campus for lunch?",
        "expected_doc": "None",
        "expected_fact": "Information not available in official documents"
    },
    {
        "id": "CAT5-MULTISTEP-PLAN",
        "category": "5. Multi-step Study Planning",
        "query": "Create a study plan for Data Structures, Operating Systems, and Computer Architecture exams on 2026-10-20 with 2 hours on weekdays and 4 hours on weekends.",
        "expected_doc": "nitte_syllabus.md / Calendar Tool",
        "expected_fact": "Day-by-day timetable generated"
    }
]

class SystemEvaluator:
    def __init__(self):
        self.pipeline = WorkflowPipeline()

    def compare_rag_vs_plain(self, query: str) -> Dict[str, Any]:
        """
        Runs query head-to-head: RAG Pipeline vs Plain LLM (without retrieval).
        """
        start_rag = time.time()
        rag_res = self.pipeline.process_user_turn(query=query, history=[])
        rag_time = round(time.time() - start_rag, 3)

        start_plain = time.time()
        llm = get_llm()
        # Use the dedicated PLAIN_LLM_TEMPLATE for true "no-context" comparison
        plain_prompt = PLAIN_LLM_TEMPLATE.format(query=query)
        try:
            plain_res = llm.invoke(plain_prompt).content
        except Exception:
            plain_res = "Plain LLM output unavailable."
        plain_time = round(time.time() - start_plain, 3)

        chunks = rag_res.get("retrieved_chunks", [])
        review = rag_res.get("review_details", {})

        return {
            "query": query,
            "rag": {
                "answer": rag_res.get("final_answer", ""),
                "latency_sec": rag_time,
                "grounded": review.get("grounded", False),
                "groundedness_score": review.get("score", 0.0),
                "review_status": review.get("status", "N/A"),
                "sources_retrieved": [c["source"] for c in chunks]
            },
            "plain_llm": {
                "answer": plain_res,
                "latency_sec": plain_time,
                "grounded": False,
                "groundedness_score": 0.0,
                "note": "Unassisted LLM generation - susceptible to generic guessing/hallucination."
            }
        }

    def run_full_benchmark(self) -> List[Dict[str, Any]]:
        """Runs evaluation suite across all 5 PRD categories."""
        results = []
        for item in PRD_BENCHMARK_SUITE:
            t0 = time.time()
            res = self.pipeline.process_user_turn(
                query=item["query"],
                history=item.get("context_history", [])
            )
            elapsed = round(time.time() - t0, 3)
            
            review = res.get("review_details", {})
            chunks = res.get("retrieved_chunks", [])

            results.append({
                "test_id": item["id"],
                "category": item["category"],
                "query": item["query"],
                "intent": res.get("intent", ""),
                "final_answer": res.get("final_answer", "")[:250] + "...",
                "latency_sec": elapsed,
                "groundedness_score": review.get("score", 0.0),
                "sources": [c["source"] for c in chunks],
                "status": "PASSED" if review.get("grounded", True) else "FAILED / FALLBACK"
            })
        return results
