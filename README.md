# 🌲 Vectorless RAG Application

A production-ready implementation of **Tree-Based Document Retrieval and Reasoning** using the [PageIndex SDK](https://pageindex.ai/) and Google's Gemini models.

This project demonstrates a Vectorless RAG architecture, replacing traditional embedding-based chunk retrieval (which often suffers from context fragmentation and hallucination) with an LLM-driven tree search. The system uses Gemini to reason over the hierarchical table-of-contents of your document to retrieve the exact sections needed for accurate, citable answers.

## ✨ Features

- **Vectorless Retrieval:** No vector databases, embeddings, or arbitrary text chunking. Respects the document's natural sections.
- **Explainable Reasoning:** The LLM's step-by-step reasoning for choosing specific document sections is fully transparent and visible in the UI.
- **Traceable Answers:** The generated answers cite the specific section titles and page numbers from the original document.
- **Domain Expert Rules:** Inject custom routing rules (e.g., "If query asks for risks, prioritize the Risk Factors section") directly via the UI without any fine-tuning.
- **Large PDF Handling:** Built-in integration with `pypdf` to extract specific page subsets, helping you navigate PageIndex free-tier size limits.
- **Session Continuity:** Directly load existing `doc_id`s to skip the upload/indexing phase for previously processed documents.
- **Real-Time Streaming:** Streams answers token-by-token directly from Gemini.

## 🛠️ Architecture & Modules

The application is modularized for clarity and maintainability:

- `app.py`: The main Streamlit interactive user interface.
- `vectorless_rag/config.py`: Centralized configuration management using `pydantic-settings`.
- `vectorless_rag/ingestion.py`: Handles uploading files to PageIndex and polling their status.
- `vectorless_rag/indexing.py`: Fetches and formats the parsed document tree.
- `vectorless_rag/retrieval.py`: Compresses the tree and performs the core LLM-driven section search.
- `vectorless_rag/generation.py`: Constructs the context window and streams grounded answers.
- `vectorless_rag/pipeline.py`: Orchestrates the retrieval and generation phases.
- `vectorless_rag/llm/gemini.py`: Isolates all Google Gemini interactions.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [PageIndex API Key](https://dash.pageindex.ai/api-keys)
- A [Google Gemini API Key](https://aistudio.google.com/)

### Installation

1. **Clone or download this repository** and navigate to the root directory.

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   
   # Activate on Windows:
   .\.venv\Scripts\activate
   # Activate on macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Create a `.env` file in the root directory (you can copy `.env.example`):

```ini
PAGEINDEX_API_KEY=your_pageindex_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional Configurations
GEMINI_MODEL=gemini-2.0-flash
PROCESSING_TIMEOUT_SECONDS=300
TREE_SUMMARY_CHARS=150
```

### Running the Application

1. **Verify your API keys** (Optional):
   Run the backend verification script to ensure your `.env` is configured correctly.
   ```bash
   python test_backend.py
   ```

2. **Start the Streamlit UI**:
   ```bash
   streamlit run app.py
   ```

3. **Interact**: 
   Open your browser to the URL provided in the terminal (usually `http://localhost:8501`). Upload a PDF (or enter an existing `doc_id`), view the extracted tree, and start chatting!

## ⚠️ Notes on PageIndex API Limits
If you are using the free tier of the PageIndex API, you may encounter `LimitReached` errors when uploading large files (e.g., > 10MB or high page counts). To work around this, check the **"Extract a subset of pages"** option in the sidebar to process a smaller portion of your document.
