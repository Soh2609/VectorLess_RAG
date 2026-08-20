import time
from typing import Generator
from pageindex import PageIndexClient
from vectorless_rag.config import settings

pi_client = PageIndexClient(api_key=settings.pageindex_api_key)

def upload_document(file_path: str) -> str:
    """
    Uploads a document to PageIndex and returns the document ID.
    """
    result = pi_client.submit_document(file_path)
    return result["doc_id"]

def wait_for_processing(doc_id: str, poll_interval_s: int = 5) -> Generator[str, None, None]:
    """
    Polls PageIndex for the processing status of a document.
    Yields status messages that can be displayed in a UI.
    Raises TimeoutError if processing exceeds settings.processing_timeout_seconds.
    """
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > settings.processing_timeout_seconds:
            raise TimeoutError(f"Document processing timed out after {settings.processing_timeout_seconds} seconds.")
            
        ready = pi_client.is_retrieval_ready(doc_id)
        
        if ready:
            yield "completed"
            break
        else:
            yield "processing"
            
        time.sleep(poll_interval_s)
