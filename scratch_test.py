import sys
sys.stdout.reconfigure(encoding='utf-8')
from backend.ingestion import ingest_directory, DEFAULT_SAMPLE_DIR
from backend.vectorstore import VectorStoreManager
from backend.workflow import WorkflowPipeline
from backend.evaluator import SystemEvaluator

print("1. Ingesting all documents from data/sample_documents...")
num_chunks = ingest_directory(DEFAULT_SAMPLE_DIR)
print(f"Ingested {num_chunks} chunks.")

print("\n2. Initializing Workflow Pipeline...")
pipeline = WorkflowPipeline()

test_queries = [
    "What is the minimum attendance required at NITTE to sit for exams?",
    "What subjects are offered in 3rd semester Cyber Security branch?",
    "What is the course code and syllabus for Data Structures in CSE?",
    "What are the modules in Ethical Hacking for ISE branch?",
    "What subjects are taught in AI and Data Science 5th semester?",
    "When is the MSE-1 exam for Discrete Mathematical Structures 25MAT204?",
    "What is the date for Unix Shell IS1602-1 practical examination?"
]

print("\n3. Testing Multi-Branch & Exam Queries...")
for q in test_queries:
    print(f"\n--- QUERY: {q} ---")
    res = pipeline.process_user_turn(query=q, history=[])
    print(f"Intent: {res['intent']}")
    review = res.get('review_details', {})
    print(f"Grounding Status: {review.get('status', 'N/A')} | Score: {review.get('score', 'N/A')}")
    ans = res.get('final_answer') or res.get('draft_answer') or str(res.get('study_plan'))
    print(f"Response: {ans[:300]}...")

print("\nAll Branch & Exam Timetable Tests Complete!")

print("\n4. Running Full 5-Category PRD Benchmark Suite...")
evaluator = SystemEvaluator()
benchmark_results = evaluator.run_full_benchmark()
passed = sum(1 for r in benchmark_results if r["status"] == "PASSED")
print(f"\nBenchmark Result: {passed}/{len(benchmark_results)} tests PASSED")
for r in benchmark_results:
    icon = "✅" if r["status"] == "PASSED" else "⚠️"
    print(f"  {icon} [{r['test_id']}] Score: {r['groundedness_score']} | Latency: {r['latency_sec']}s | {r['category']}")

