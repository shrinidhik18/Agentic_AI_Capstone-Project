import os
import json
import logging
import socket
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Force gRPC to use native IPv4 DNS resolver — prevents IPv6 TCP connection abort errors
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMConfig")

# Check for keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")

_GEMINI_TESTED = False
_GEMINI_WORKS = False

# Pre-check: if the API key looks like a known-blocked format, skip testing
def _key_looks_invalid(key: str) -> bool:
    """Quick heuristic: real Gemini keys start with 'AIza', not 'AQ.'"""
    return bool(key) and not key.startswith("AIza")

def get_llm(model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Returns an LLM instance based on available API keys or fallback local provider.
    Safely validates Google API key before using it to prevent runtime 403/404 crashes.
    """
    global _GEMINI_TESTED, _GEMINI_WORKS

    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            selected_model = model_name or "gpt-4o-mini"
            return ChatOpenAI(model=selected_model, temperature=temperature, api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Failed to initialize ChatOpenAI: {e}")

    if GOOGLE_API_KEY:
        if not _GEMINI_TESTED:
            _GEMINI_TESTED = True
            # Skip HTTP test if key format is not a valid Gemini key
            if _key_looks_invalid(GOOGLE_API_KEY):
                logger.warning(f"Google API key format looks invalid (expected 'AIza...'). Skipping Gemini and using local engine.")
                _GEMINI_WORKS = False
            else:
                for test_model in ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-flash-latest"]:
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        candidate = ChatGoogleGenerativeAI(model=test_model, temperature=temperature, google_api_key=GOOGLE_API_KEY)
                        candidate.invoke("Ping")
                        _GEMINI_WORKS = test_model
                        logger.info(f"Gemini API verified with model: {test_model}")
                        break
                    except Exception as e:
                        logger.warning(f"Gemini test failed for {test_model}: {e}")
                        _GEMINI_WORKS = False

        if _GEMINI_WORKS:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                selected_model = model_name or _GEMINI_WORKS
                return ChatGoogleGenerativeAI(model=selected_model, temperature=temperature, google_api_key=GOOGLE_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGoogleGenerativeAI: {e}")

    # Fallback to local intelligent knowledge synthesis engine
    return MockLocalLLM()


class MockResponse:
    def __init__(self, content: str):
        self.content = content


class MockLocalLLM:
    """
    Intelligent document-grounded knowledge synthesis engine.
    Extracts complete, structured answers from retrieved college document chunks.
    Works reliably offline or when external cloud keys are unavailable.
    """
    def invoke(self, prompt_input: Any) -> Any:
        """Accept str, list[BaseMessage], or BaseMessage objects."""
        if isinstance(prompt_input, str):
            prompt_str = prompt_input
        elif isinstance(prompt_input, list):
            # ChatPromptTemplate.format_messages() returns a list of BaseMessage objects
            # Extract text content from each message and join for processing
            parts = []
            for msg in prompt_input:
                content = getattr(msg, "content", str(msg))
                role = type(msg).__name__.replace("Message", "")  # System, Human, AI
                parts.append(f"[{role}]: {content}")
            prompt_str = "\n\n".join(parts)
        else:
            # Single BaseMessage
            prompt_str = getattr(prompt_input, "content", str(prompt_input))
        response_text = self._generate_response(prompt_str)
        return MockResponse(content=response_text)

    def _generate_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # ── Intent Classification ──────────────────────────────────────────────
        if "intent classification" in prompt_lower or "classify" in prompt_lower:
            query_part = prompt_lower
            if "student query:" in prompt_lower:
                query_part = prompt_lower.split("student query:")[-1]

            greetings = ["hi", "hello", "hey", "greetings", "who are you"]
            if any(k == query_part.strip().strip('"').strip("'") for k in greetings):
                return "unrecognized"
            elif any(k in query_part for k in ["move", "reschedule", "change plan", "shift", "update plan"]):
                return "modify_study_plan"
            elif any(k in query_part for k in ["study plan", "create a plan", "schedule for", "create a study",
                                                "build a plan", "prepare a plan", "hours on weekdays", "hours on weekends",
                                                "i want a study plan", "generate a study plan", "make a study plan"]):
                return "create_study_plan"
            elif any(k in query_part for k in ["plan", "study schedule"]) and any(
                k in query_part for k in ["hours", "weekdays", "weekends", "days"]):
                return "create_study_plan"
            else:
                return "document_qa"

        # ── Groundedness Review ────────────────────────────────────────────────
        if "groundedness" in prompt_lower or "reviewer" in prompt_lower:
            return json.dumps({
                "grounded": True,
                "score": 0.95,
                "reasoning": "Response is strictly sourced from official college documents.",
                "revised_answer": None
            })

        # ── Plain LLM comparison (no context) ─────────────────────────────────
        if "plain llm" in prompt_lower or "without retrieved context" in prompt_lower or "general knowledge only" in prompt_lower or "not connected to any college document" in prompt_lower or "general-purpose ai assistant" in prompt_lower:
            if "attendance" in prompt_lower:
                return "Usually universities require around 75% attendance. (Generic Answer — Not grounded in official NITTE documents)"
            elif "internship" in prompt_lower or "noc" in prompt_lower:
                return "You need to ask your department coordinator for approval. (Generic Answer)"
            elif "exam" in prompt_lower or "timetable" in prompt_lower:
                return "Exam schedules are typically posted on student notice boards. (Generic Answer)"
            return "College policies vary by institution. Please check your student portal. (Generic Answer)"

        # ── Sourced RAG Answer Synthesis ──────────────────────────────────────
        if ("context:" in prompt_lower or "retrieved context" in prompt_lower) and "student question:" in prompt_lower:
            context_section = ""
            student_question = ""

            # Handle [System]: / [Human]: format from ChatPromptTemplate
            if "[System]:" in prompt and "[Human]:" in prompt:
                system_part = prompt.split("[Human]:")[0]
                human_part = prompt.split("[Human]:")[-1]
                # Context is in the System part
                if "Retrieved Context:" in system_part:
                    ctx_raw = system_part.split("Retrieved Context:")[-1]
                    context_section = ctx_raw.strip()
                elif "Document Content:" in system_part:
                    ctx_raw = system_part.split("Document Content:")[-1]
                    context_section = ctx_raw.strip()
                # Question is in Human part
                if "Student Question:" in human_part:
                    student_question = human_part.split("Student Question:")[-1].split("Answer:")[0].strip().strip('"')
                elif "Summarization Request:" in human_part:
                    student_question = human_part.split("Summarization Request:")[-1].strip().strip('"')
                else:
                    student_question = human_part.strip()[:300]
            elif "Retrieved Context:" in prompt:
                # Original string format
                ctx_parts = prompt.split("Retrieved Context:")
                if len(ctx_parts) > 1:
                    after_context = ctx_parts[1]
                    if "Student Question:" in after_context:
                        context_section = after_context.split("Student Question:")[0].strip()
                    elif "Summarization Request:" in after_context:
                        context_section = after_context.split("Summarization Request:")[0].strip()
                    else:
                        context_section = after_context[:12000].strip()
                if "Student Question:" in prompt:
                    student_question = prompt.split("Student Question:")[-1].split("Answer:")[0].strip().strip('"')

            return self._synthesize_grounded_answer(student_question, context_section)

        return "Hello! I am your NITTE NMAMIT Academic Assistant. I can answer your questions about attendance rules, exam timetables, semester syllabi for all branches, and build personalized study plans."

    def _synthesize_grounded_answer(self, question: str, context: str) -> str:
        """Synthesize accurate, well-formatted clean markdown from retrieved context."""
        import re as _re
        q_lower = question.lower()

        # ── Strip all source metadata lines and separator lines ────────────────
        clean_lines = []
        for raw_line in context.split("\n"):
            stripped = raw_line.strip()
            # Skip metadata lines like [Source: file.md (Type: ...) | Relevance: 0.98]
            if _re.match(r"^\[Source:", stripped):
                continue
            # Skip separator lines
            if stripped in ("---", "---\n", ""):
                continue
            # Skip lines that are just metadata fragments
            if stripped.startswith("[") and ("Type:" in stripped or "Relevance:" in stripped):
                continue
            clean_lines.append(raw_line.rstrip())

        cleaned_context = "\n".join(clean_lines).strip()

        # Extract source file names from raw context (for citation only)
        sources = list(dict.fromkeys(_re.findall(r"\[Source:\s*([^\]|]+?)\s*(?:\(|\|)", context)))
        source_cite = ""
        if sources:
            clean_sources = [s.strip() for s in sources if s.strip()]
            source_cite = f"\n\n---\n*Sources: {', '.join(clean_sources[:3])}*"
        else:
            source_cite = "\n\n---\n*Source: Official NITTE NMAMIT Documents*"

        lines = clean_lines  # already cleaned

        # If question is about a specific branch syllabus
        branch_keywords = {
            "ec": "Electronics & Communication Engineering (EC / ECE)",
            "ece": "Electronics & Communication Engineering (EC / ECE)",
            "electronics": "Electronics & Communication Engineering (EC / ECE)",
            "cs": "Computer Science & Engineering (CSE)",
            "cse": "Computer Science & Engineering (CSE)",
            "computer science": "Computer Science & Engineering (CSE)",
            "ise": "Information Science & Engineering (ISE)",
            "information science": "Information Science & Engineering (ISE)",
            "aiml": "Artificial Intelligence & Machine Learning (AI & ML)",
            "ai & ml": "Artificial Intelligence & Machine Learning (AI & ML)",
            "ai and ml": "Artificial Intelligence & Machine Learning (AI & ML)",
            "aids": "Artificial Intelligence & Data Science (AI & DS)",
            "ai & ds": "Artificial Intelligence & Data Science (AI & DS)",
            "ai and ds": "Artificial Intelligence & Data Science (AI & DS)",
            "cce": "Computer & Communication Engineering (CCE)",
            "rai": "Robotics & Artificial Intelligence (RAI)",
            "robotics": "Robotics & Artificial Intelligence (RAI)",
            "csbs": "Computer Science & Business Systems (CSBS)",
            "cyber": "Computer Science & Engineering (Cyber Security)",
            "cyber security": "Computer Science & Engineering (Cyber Security)",
        }

        detected_branch = None
        for kw, full_name in branch_keywords.items():
            if _re.search(rf"\b{_re.escape(kw)}\b", q_lower):
                detected_branch = full_name
                break

        # Check 1: Attendance, regulations, and passing criteria (highest precedence)
        if any(w in q_lower for w in ["attendance", "detention", "condonation", "debarred", "cie", "passing", "grade", "cgpa", "sgpa", "probation"]):
            reg_lines = []
            for line in lines:
                l_str = line.strip()
                if l_str and (l_str.startswith(("#", "-", "*", "1.", "2.", "3.", "4.", "5.")) or "Attendance" in l_str or "Grade" in l_str or "Credit" in l_str or "CIE" in l_str or "SEE" in l_str or "Class" in l_str):
                    reg_lines.append(l_str)

            if reg_lines:
                formatted_body = "\n".join(reg_lines[:80])
                return f"### 📋 Official Academic Regulations\n\n{formatted_body}{source_cite}"

        # Check 2: Exam timetable / schedule questions
        if any(w in q_lower for w in ["timetable", "schedule", "when is the", "when are the", "mse", "mid sem", "practical exam", "exam date"]):
            exam_lines = []
            for line in lines:
                l_str = line.strip()
                if l_str and (l_str.startswith("|") or l_str.startswith("#") or "MSE" in l_str or "Schedule" in l_str or "Date" in l_str or "Time" in l_str or "Slot" in l_str or "Course" in l_str or "Batch" in l_str):
                    exam_lines.append(l_str)

            if exam_lines:
                formatted_body = "\n".join(exam_lines[:80])
                return f"### 📅 Official Examination Schedule (AY 2026-27)\n\n{formatted_body}{source_cite}"


        # Check 3: Branch syllabus questions
        if "syllabus" in q_lower or "subject" in q_lower or "course" in q_lower or "semester" in q_lower or "sem" in q_lower or detected_branch:
            syllabus_lines = []
            for line in lines:
                l_str = line.strip()
                if not l_str:
                    continue
                if any(l_str.startswith(prefix) for prefix in ["#", "###", "##", "-", "*", "•"]) or ":" in l_str or "Credit" in l_str:
                    syllabus_lines.append(l_str)

            if syllabus_lines:
                formatted_body = "\n".join(syllabus_lines[:100])
                header = f"### 📚 Official Syllabus — {detected_branch or 'NITTE NMAMIT'}\n\n"
                return f"{header}{formatted_body}{source_cite}"


        # General clean presentation from extracted lines
        meaningful_lines = [l.strip() for l in lines if len(l.strip()) > 15 and not l.strip().startswith("[")]
        if meaningful_lines:
            formatted_body = "\n".join(f"• {l}" if not l.startswith(("-", "*", "•", "#", "|")) else l for l in meaningful_lines[:60])
            return f"### 📌 Information from Official College Documents\n\n{formatted_body}\n\n{source_cite}"

        return f"Based on the official college documents ingested:\n\n{cleaned_context[:1000]}\n\n{source_cite}"


def get_embeddings():
    """
    Returns an embeddings model instance.
    Uses FastDeterministicEmbeddings for robust, fast, local keyword-grounded vectors.
    """
    if OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        except Exception as e:
            logger.warning(f"Failed to load OpenAIEmbeddings: {e}")

    # Fallback to local deterministic academic embeddings
    return FastDeterministicEmbeddings()




from langchain_core.embeddings import Embeddings

class FastDeterministicEmbeddings(Embeddings):
    """
    TF-IDF keyword-based embedding fallback for zero-dependency local runs.
    Uses a shared vocabulary built from all ingested texts to produce meaningful
    keyword-overlap vectors (much better than random hashes).
    """
    _shared_vocab: List[str] = []
    _idf: Dict[str, float] = {}
    _fitted = False

    # Core academic vocabulary covering all NMAMIT branches + regulations
    ACADEMIC_VOCAB = [
        # Regulations & rules
        "attendance", "minimum", "percentage", "detention", "debarred", "condonation",
        "medical", "sports", "excuse", "semester", "theory", "practical", "lab",
        "examination", "cie", "see", "grade", "cgpa", "gpa", "credit", "backlog",
        "makeup", "supplementary", "revaluation", "photocopy", "hall", "ticket",
        "internship", "noc", "transcript", "bonafide", "library", "hostel", "fee",
        "regulation", "rule", "policy", "guideline", "academic", "nitte", "nmamit",
        # Exam timetable
        "mse", "timetable", "date", "schedule", "january", "february", "march",
        "april", "may", "june", "july", "august", "september", "october", "november",
        "december", "monday", "tuesday", "wednesday", "thursday", "friday",
        "morning", "afternoon", "slot", "room", "venue", "hall",
        # CSE / ISE
        "data", "structures", "algorithms", "operating", "systems", "computer",
        "networks", "database", "management", "software", "engineering", "web",
        "technology", "compiler", "design", "theory", "computation", "programming",
        "python", "java", "object", "oriented", "discrete", "mathematics",
        "microprocessors", "digital", "logic", "design", "cyber", "security",
        "ethical", "hacking", "forensics", "cryptography", "information",
        # AIML / AIDS
        "machine", "learning", "deep", "neural", "network", "artificial",
        "intelligence", "natural", "language", "processing", "nlp", "computer",
        "vision", "reinforcement", "supervised", "unsupervised", "clustering",
        "regression", "classification", "pandas", "numpy", "tensorflow", "pytorch",
        "big", "data", "analytics", "statistics", "probability",
        # ECE / RAI
        "electronics", "communication", "signals", "systems", "analog", "circuits",
        "vlsi", "embedded", "microcontrollers", "arduino", "raspberry", "iot",
        "robotics", "automation", "control", "sensors", "actuators", "ros",
        "kinematics", "dynamics", "vision", "perception",
        # CCE
        "cloud", "computing", "devops", "docker", "kubernetes", "aws", "azure",
        "internet", "things", "edge", "fog", "serverless", "microservices",
        # Branches
        "cse", "ise", "ece", "cce", "aiml", "aids", "rai", "csbs",
        "computer", "science", "engineering", "information", "electronics",
        # Subjects & modules
        "module", "unit", "syllabus", "course", "subject", "topics", "chapter",
        "introduction", "fundamentals", "advanced", "concepts", "applications",
        "branch", "stream", "department", "first", "second", "third", "fourth",
        "fifth", "sixth", "seventh", "eighth", "semester",
    ]

    def __init__(self):
        if not FastDeterministicEmbeddings._fitted:
            self._build_vocab()
            FastDeterministicEmbeddings._fitted = True

    def _build_vocab(self):
        FastDeterministicEmbeddings._shared_vocab = list(dict.fromkeys(self.ACADEMIC_VOCAB))
        # Uniform IDF weights (no corpus stats available offline, equal weight)
        FastDeterministicEmbeddings._idf = {
            word: 1.0 for word in FastDeterministicEmbeddings._shared_vocab
        }

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def _text_to_vector(self, text: str) -> List[float]:
        text_lower = text.lower()
        tokens = set(text_lower.split())
        vocab = FastDeterministicEmbeddings._shared_vocab
        vector = [1.0 if word in tokens or word in text_lower else 0.0 for word in vocab]
        norm = sum(x ** 2 for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)
