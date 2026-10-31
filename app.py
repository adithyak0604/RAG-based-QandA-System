"""
Streamlit GUI for the RAG Q&A system.

Run with:
    streamlit run app.py

Sidebar navigation switches between:
- Quiz App: the 3-step flow (Upload & Ingest / Generate Questions / Answer Questions)
- History: browse previously generated question sets, grouped by document

Every generated question set is auto-saved to storage/question_store.py
(SQLite) so it survives app restarts, instead of vanishing with the
Streamlit session.

Note: audio answers aren't wired in yet (next stage) — this is text-only
for now.
"""
import os

import streamlit as st

from ingestion.loaders import load_document
from processing.chunking import chunk_document
from storage.vector_store import add_chunks, list_source_files, get_chunk_by_id, delete_source
from storage.question_store import save_question_set, list_question_sets, get_question_set, delete_question_set
from generate_questions_pipeline import generate_question_set
from config import settings

st.set_page_config(page_title="RAG Q&A System", layout="wide")

# Streamlit's st.selectbox is built on a searchable combobox (BaseWeb's
# Select component), so its clickable area is technically a text <input>
# under the hood — browsers always show a text-caret cursor over any
# <input>, which is why it looks like a text field instead of a plain
# dropdown. This CSS overrides just the cursor style to a pointer,
# without changing the widget's actual (still searchable) behavior.
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("📚 RAG Q&A System")


def show_reference(q: dict, key_suffix: str):
    """Renders an expandable 'Reference' section showing exactly which
    part of the uploaded document a question came from."""
    with st.expander(f"📄 Reference — {q['source_file']}, page {q['page_number']}", expanded=False):
        chunk = get_chunk_by_id(q["source_chunk_id"])
        if chunk:
            st.markdown(chunk["text"])
        else:
            st.caption("Source text not found (it may have been re-ingested since this question was generated).")

with st.sidebar:
    st.subheader("Settings")

    if settings.env == "production":
        # Ollama runs on a local machine and isn't reachable from a
        # deployed server — don't offer a choice that would just crash.
        settings.llm_backend_override = "openrouter"
        st.caption("LLM backend: `openrouter` (fixed in production)")
    else:
        with st.expander("🧠 LLM Choice", expanded=False):
            backend_choice = st.radio(
                "Choose LLM backend",
                options=["Auto (recommended)", "Ollama (local, free, slower)", "OpenRouter (API, faster, costs money)"],
                index=0,
            )
            if backend_choice.startswith("Auto"):
                settings.llm_backend_override = "auto"
            elif backend_choice.startswith("Ollama"):
                settings.llm_backend_override = "ollama"
            else:
                settings.llm_backend_override = "openrouter"

            st.caption(f"Current: `{settings.llm_backend_override}`")

    st.divider()
    page = st.selectbox("📂 Navigate", ["Quiz App", "History"])


if page == "Quiz App":
    tab1, tab2, tab3 = st.tabs(["1. Upload & Ingest", "2. Generate Questions", "3. Answer Questions"])


    # ---------------------------------------------------------------------
    # Tab 1: Upload & Ingest
    # ---------------------------------------------------------------------
    with tab1:
        st.header("Upload a document")
        uploaded_file = st.file_uploader(
            "Supported formats: PDF, DOCX, PPTX, TXT",
            type=["pdf", "docx", "pptx", "txt"],
        )

        if uploaded_file is not None:
            filepath = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Ingest this document"):
                with st.spinner("Loading document..."):
                    pages = load_document(filepath)
                st.write(f"Extracted {len(pages)} page(s)/section(s)")

                with st.spinner("Chunking..."):
                    chunks = chunk_document(
                        pages,
                        chunk_size_tokens=settings.chunk_size_tokens,
                        overlap_tokens=settings.chunk_overlap_tokens,
                    )
                st.write(f"Created {len(chunks)} chunk(s)")

                with st.spinner("Embedding + storing (first run downloads the model)..."):
                    add_chunks(chunks)
                st.success(f"'{uploaded_file.name}' ingested and ready for question generation.")

        st.divider()
        st.subheader("Already ingested documents")
        existing = list_source_files()
        if existing:
            for doc_name in existing:
                col_name, col_button = st.columns([5, 1])
                with col_name:
                    st.markdown(f"📄 {doc_name}")
                with col_button:
                    if st.button("🗑️ Delete", key=f"delete_{doc_name}"):
                        n_deleted = delete_source(doc_name)
                        st.toast(f"Deleted {doc_name} ({n_deleted} chunks removed)")
                        st.rerun()
        else:
            st.caption("Nothing ingested yet.")


    # ---------------------------------------------------------------------
    # Tab 2: Generate Questions
    # ---------------------------------------------------------------------
    with tab2:
        st.header("Generate questions from an ingested document")
        available_docs = list_source_files()

        if not available_docs:
            st.warning("Ingest a document in Tab 1 first.")
        else:
            selected_doc = st.selectbox("Choose a document", available_docs)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_mcq = st.number_input("MCQ", min_value=0, max_value=10, value=2)
            with col2:
                n_msq = st.number_input("MSQ", min_value=0, max_value=10, value=1)
            with col3:
                n_obj = st.number_input("Objective", min_value=0, max_value=10, value=2)
            with col4:
                n_subj = st.number_input("Subjective", min_value=0, max_value=10, value=1)

            if st.button("Generate questions"):
                counts = {"MCQ": n_mcq, "MSQ": n_msq, "Objective": n_obj, "Subjective": n_subj}
                with st.spinner("Generating questions via LLM... this may take a minute"):
                    questions = generate_question_set(selected_doc, counts)
                save_question_set(selected_doc, questions)
                st.session_state["questions"] = questions
                st.session_state["answers"] = {}
                st.session_state["quiz_result"] = None
                st.success(f"Generated {len(questions)} question(s) and saved to History. Go to Tab 3 to answer them.")

            if "questions" in st.session_state:
                st.divider()
                st.subheader("Preview")
                for i, q in enumerate(st.session_state["questions"], start=1):
                    st.markdown(f"**{i}. [{q['type']}]** {q['question']}")
                    show_reference(q, key_suffix=f"preview_{i}")


    # ---------------------------------------------------------------------
    # Tab 3: Answer Questions
    # ---------------------------------------------------------------------
    with tab3:
        st.header("Answer the generated questions")

        if "questions" not in st.session_state or not st.session_state["questions"]:
            st.warning("Generate questions in Tab 2 first.")
        else:
            if "answers" not in st.session_state:
                st.session_state["answers"] = {}

            # Once submitted, lock in and show results instead of the answer form.
            if st.session_state.get("quiz_result"):
                result = st.session_state["quiz_result"]
                st.subheader(f"Score: {result['total_score']} / {result['max_score']} ({result['percentage']}%)")

                for i, r in enumerate(result["results"]):
                    icon = "✅" if r["correct"] else ("🟡" if r["score"] > 0 else "❌")
                    st.markdown(f"---\n**{icon} Q{i+1} [{r['type']}]:** {r['question']}")
                    st.write(f"Your answer: {r['user_answer']}")
                    st.caption(r["feedback"])
                    show_reference(r, key_suffix=f"result_{i}")

                if st.button("Retake with new answers"):
                    st.session_state["quiz_result"] = None
                    st.session_state["answers"] = {}
                    st.rerun()

            else:
                for i, q in enumerate(st.session_state["questions"]):
                    st.markdown(f"---\n**Q{i+1} [{q['type']}]:** {q['question']}")
                    show_reference(q, key_suffix=f"answer_{i}")

                    if q["type"] == "MCQ":
                        answer = st.radio("Select one:", q["options"], key=f"q_{i}", index=None)
                    elif q["type"] == "MSQ":
                        answer = []
                        for opt in q["options"]:
                            if st.checkbox(opt, key=f"q_{i}_{opt}"):
                                answer.append(opt)
                    elif q["type"] == "Objective":
                        answer = st.text_input("Your answer:", key=f"q_{i}")
                    else:  # Subjective
                        answer = st.text_area("Your answer:", key=f"q_{i}", height=120)

                    st.session_state["answers"][i] = answer

                st.divider()
                if st.button("Submit Quiz", type="primary"):
                    from processing.evaluation import evaluate_quiz
                    with st.spinner("Grading your answers..."):
                        result = evaluate_quiz(st.session_state["questions"], st.session_state["answers"])
                    st.session_state["quiz_result"] = result
                    st.rerun()

# ---------------------------------------------------------------------
# History page: browse previously generated question sets, per document
# ---------------------------------------------------------------------
else:
    st.header("Question Generation History")

    all_sets = list_question_sets()
    if not all_sets:
        st.caption("No question sets generated yet. Go to Quiz App -> Generate Questions.")
    else:
        docs_with_history = sorted({s["source_file"] for s in all_sets})

        for doc_name in docs_with_history:
            st.subheader(f"📄 {doc_name}")
            doc_sets = list_question_sets(doc_name)

            for s in doc_sets:
                counts_str = ", ".join(f"{k}: {v}" for k, v in s["counts"].items())
                col_info, col_load, col_delete = st.columns([5, 2, 1])

                with col_info:
                    st.write(f"{s['created_at'][:19].replace('T', ' ')} UTC — {s['total']} question(s) ({counts_str})")
                with col_load:
                    if st.button("Load into Quiz", key=f"load_{s['id']}"):
                        questions = get_question_set(s["id"])
                        st.session_state["questions"] = questions
                        st.session_state["answers"] = {}
                        st.session_state["quiz_result"] = None
                        st.toast("Loaded — switch to 'Quiz App' in the sidebar, then Tab 3 to answer.")
                with col_delete:
                    if st.button("🗑️", key=f"delhist_{s['id']}"):
                        delete_question_set(s["id"])
                        st.rerun()

            st.divider()
