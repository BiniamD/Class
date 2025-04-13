import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Get the number of pages
        num_pages = len(pdf_reader.pages)
        
        # Extract text from all pages
        all_text = []
        for page_num in range(num_pages):
            # Get the page
            page = pdf_reader.pages[page_num]
            
            # Extract text from the page
            text = page.extract_text()
            if text:
                all_text.append(text)
        
        return all_text

if __name__ == "__main__":
    pdf_path = "FinancialIQ/documents/850ce1be-86eb-4563-9ce7-fbe104826dc5_Untitled.pdf"
    text_content = extract_text_from_pdf(pdf_path)
    
    # Print each page's content
    for i, text in enumerate(text_content, 1):
        print(f"\nPage {i}:")
        print("-" * 50)
        print(text)
        print("-" * 50) 