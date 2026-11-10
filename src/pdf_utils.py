from pypdf import PdfReader

def get_paper_text(uploaded_file) -> str:
    """Extracts and concatenates text from all pages of a PDF file object."""
    reader = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
