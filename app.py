import os
import streamlit as st
import tempfile
import pypdf
from vectorless_rag.ingestion import upload_document, wait_for_processing
from vectorless_rag.indexing import fetch_tree, format_tree_text
from vectorless_rag.pipeline import run_rag
from vectorless_rag.llm import get_active_provider_info

# Must be the first Streamlit command
st.set_page_config(
    page_title="Vectorless RAG",
    page_icon="🌲",
    layout="wide"
)

def init_session_state():
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    init_session_state()
    
    st.title("🌲 Vectorless RAG Application Using PageIndex")
    st.markdown("A production-ready implementation of tree-based document retrieval and reasoning using Open source groq model.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        try:
            from vectorless_rag.config import settings
            st.success("✅ Configuration Loaded")
            provider_info = get_active_provider_info()
            status_emoji = "🟢" if provider_info["status"] == "primary" else "🔵"
            st.markdown(f"**LLM Provider:** {status_emoji} {provider_info['name']}")
            st.caption(f"Model: {provider_info['model']}")
        except Exception as e:
            st.error(f"❌ Configuration Error: {e}")
            st.stop()
            
        st.divider()
        st.header("📄 Document Management")
        
        # Load existing document ID
        existing_doc_id = st.text_input("Enter existing Document ID (optional)")
        if st.button("Load Existing Document"):
            if existing_doc_id:
                try:
                    with st.status(f"Loading {existing_doc_id}...", expanded=True) as status:
                        st.write("🌲 Fetching tree structure...")
                        tree = fetch_tree(existing_doc_id)
                        st.session_state.doc_id = existing_doc_id
                        st.session_state.tree = tree
                        st.session_state.chat_history = []
                        status.update(label="Document loaded successfully!", state="complete", expanded=False)
                except Exception as e:
                    st.error(f"Failed to load document: {e}")
            else:
                st.warning("Please enter a Document ID first.")
                
        st.markdown("**OR** Upload New PDF:")
        
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file:
            MAX_MB = settings.max_upload_size_mb
            if uploaded_file.size > MAX_MB * 1024 * 1024:
                st.error(f"❌ File too large! Max allowed: {MAX_MB}MB. Your file: {uploaded_file.size / 1e6:.1f}MB")
                st.stop()
        
        # Add a page range selector for large files
        extract_pages = st.checkbox("Extract a subset of pages (useful for large files/API limits)")
        page_start, page_end = 1, 10
        if extract_pages:
            col1, col2 = st.columns(2)
            page_start = col1.number_input("Start Page", min_value=1, value=1)
            page_end = col2.number_input("End Page", min_value=1, value=10)
            
        if uploaded_file and st.button("Process Document", type="primary"):
            with st.status("Processing Document...", expanded=True) as status:
                try:
                    # Save uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        if extract_pages:
                            # Use pypdf to extract the subset
                            reader = pypdf.PdfReader(uploaded_file)
                            writer = pypdf.PdfWriter()
                            
                            # pypdf uses 0-based indexing
                            start_idx = max(0, page_start - 1)
                            end_idx = min(len(reader.pages), page_end)
                            
                            for i in range(start_idx, end_idx):
                                writer.add_page(reader.pages[i])
                            
                            writer.write(tmp_file)
                        else:
                            tmp_file.write(uploaded_file.getvalue())
                            
                        tmp_path = tmp_file.name
                        
                    st.write("📤 Uploading to PageIndex...")
                    doc_id = upload_document(tmp_path)
                    st.session_state.doc_id = doc_id
                    st.write(f"✅ Uploaded (ID: {doc_id})")
                    
                    st.write("⏳ Waiting for tree index generation...")
                    for state in wait_for_processing(doc_id):
                        if state == "completed":
                            st.write("✅ Tree index built!")
                        elif state == "failed":
                            st.error("❌ Processing failed.")
                            st.stop()
                    
                    st.write("🌲 Fetching tree structure...")
                    tree = fetch_tree(doc_id)
                    st.session_state.tree = tree
                    
                    status.update(label="Document processed successfully!", state="complete", expanded=False)
                    # Clean up temp file
                    os.unlink(tmp_path)
                except Exception as e:
                    error_msg = str(e)
                    if "LimitReached" in error_msg:
                        status.update(label="API Limit Reached: File is too large or quota exceeded.", state="error")
                        st.error("The PageIndex API returned a 'LimitReached' error. This usually means the PDF is too large for your current API plan (e.g., > 10MB or page limit). Try enabling 'Extract a subset of pages' above to test a smaller portion of the document, or upgrade your PageIndex plan.")
                    else:
                        status.update(label=f"Error: {error_msg}", state="error")
                    
        if st.session_state.doc_id:
            st.info(f"💾 **Save this ID to resume your session later:** `{st.session_state.doc_id}`")
            st.success(f"Active Document: {st.session_state.doc_id}")
            if st.button("Clear Document"):
                st.session_state.doc_id = None
                st.session_state.tree = None
                st.session_state.chat_history = []
                st.rerun()
                
    # Main Content Area
    if not st.session_state.tree:
        st.info("👈 Please upload and process a document to begin.")
        return
        
    # Tree Inspector
    with st.expander("🌲 View Document Tree Structure"):
        st.text(format_tree_text(st.session_state.tree))
        
    st.divider()
    
    # Optional Expert Rules
    with st.expander("🛠️ Domain Expert Rules (Optional)"):
        st.markdown("Add custom routing rules to guide the LLM's retrieval. For example:")
        st.code("- If query asks for risks -> prioritize Risk Factors section\n- If query asks for revenue -> prioritize Financial Statements")
        expert_rules = st.text_area(
            "Custom Rules", 
            value="", 
            placeholder="Enter rules here to inject domain expertise...",
            height=150
        )
        
    st.header("💬 Query Document")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("reasoning"):
                with st.expander("🧠 Retrieval Reasoning & Sources"):
                    st.markdown("**Reasoning:**")
                    st.write(msg["reasoning"])
                    st.markdown("**Retrieved Sections:**")
                    for section in msg["sections"]:
                        st.write(f"- {section}")
            
    # Query input
    if query := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching document tree..."):
                try:
                    rag_result = run_rag(
                        query=query, 
                        tree=st.session_state.tree,
                        expert_rules=expert_rules if expert_rules.strip() else None
                    )
                    
                    # Display the reasoning in an expander
                    with st.expander("🧠 Retrieval Reasoning & Sources", expanded=True):
                        st.markdown("**Reasoning:**")
                        st.write(rag_result.reasoning)
                        st.markdown("**Retrieved Sections:**")
                        if rag_result.sections:
                            for section in rag_result.sections:
                                st.write(f"- {section}")
                        else:
                            st.write("None")
                    
                    # Stream the answer
                    st.markdown("**Answer:**")
                    answer = st.write_stream(rag_result.answer_generator)
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "reasoning": rag_result.reasoning,
                        "sections": rag_result.sections
                    })
                except Exception as e:
                    st.error(f"Error during retrieval: {e}")

if __name__ == "__main__":
    main()
