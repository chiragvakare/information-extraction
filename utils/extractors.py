import requests
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

def extract_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from the parsed HTML
        text_content = soup.get_text(separator='\n', strip=True)
        
        return text_content

    except requests.RequestException as e:
        return f"An error occurred: {e}"

def extract_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"An error occurred: {e}"

def extract_from_text(uploaded_file):
    try:
        # Try to read and decode the file in UTF-8
        content = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Fallback to read and decode the file in UTF-16
            content = uploaded_file.read().decode("utf-16")
        except UnicodeDecodeError:
            return "The file could not be decoded. Please check the file encoding."
    except Exception as e:
        return f"An error occurred: {e}"
    
    return content


def chunk_content(content, chunk_size=500):
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
