import pdfplumber
import re

def extract_text_from_pdf(pdf_source):
    """
    Extracts all text from a given PDF file using pdfplumber.
    """
    try:
        full_text = ""
        with pdfplumber.open(pdf_source) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def extract_section(text, section_identifier="6.3.1"):
    """
    Extracts the text corresponding to a specific section from the full text.
    """
    pattern = re.compile(
        rf"({re.escape(section_identifier)}[\s\S]*?)(?=6\.3\.2|$)",
        re.DOTALL
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return f"Section {section_identifier} not found in the document."

def get_machine_procedure(pdf_source, machine_name, section_identifier="6.3.1"):
    """
    Extracts only the specified section from the SOP PDF.
    """
    full_text = extract_text_from_pdf(pdf_source)
    if not full_text:
        return "Failed to extract SOP text."
    return extract_section(full_text, section_identifier)