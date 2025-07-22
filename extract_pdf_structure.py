import os
import json
import re
from pathlib import Path
import fitz  # PyMuPDF
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStructureExtractor:
    def __init__(self):
        self.heading_patterns = [
            # Common heading patterns
            r'^(\d+\.?\s+[A-Z][^.]*?)$',  # "1. Introduction" or "1 Introduction"
            r'^([A-Z][A-Z\s]{2,}?)$',     # "INTRODUCTION" (all caps)
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*?)$',  # "Title Case Heading"
            r'^(Chapter\s+\d+[:\s-]*.*)$',  # "Chapter 1: Title"
            r'^(Section\s+\d+[:\s-]*.*)$', # "Section 1: Title"
            r'^([IVX]+\.?\s+[A-Z][^.]*?)$', # Roman numerals "I. Introduction"
            r'^([a-z]\)\s+[A-Z][^.]*?)$',   # "a) Subsection"
            r'^([A-Z]\.\s+[A-Z][^.]*?)$',   # "A. Main Section"
        ]
   
    def extract_text_with_formatting(self, pdf_path):
        """Extract text with font size and style information"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            page_data = {
                'page_num': page_num + 1,
                'lines': []
            }
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        font_sizes = []
                        font_flags = []
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                                font_sizes.append(span["size"])
                                font_flags.append(span["flags"])
                        
                        line_text = line_text.strip()
                        if line_text and len(line_text) > 2:  # Filter out very short lines
                            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                            is_bold = any(flag & 2**4 for flag in font_flags)  # Bold flag
                            
                            page_data['lines'].append({
                                'text': line_text,
                                'font_size': avg_font_size,
                                'is_bold': is_bold,
                                'bbox': line["bbox"]
                            })
            
            pages_data.append(page_data)
        
        doc.close()
        return pages_data
   
    def extract_title(self, pages_data):
        """Extract document title from first few pages"""
        title_candidates = []
        
        # Look at first 3 pages for title
        for page in pages_data[:3]:
            for line in page['lines'][:10]:  # First 10 lines of each page
                text = line['text'].strip()
                if len(text) > 5 and len(text) < 100:  # Reasonable title length
                    # Higher score for larger fonts and bold text
                    score = line['font_size']
                    if line['is_bold']:
                        score += 5
                    if page['page_num'] == 1:  # First page bonus
                        score += 3
                    
                    title_candidates.append((text, score))
        
        if title_candidates:
            # Sort by score and return the highest
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            return title_candidates[0][0]
        
        return "Untitled Document"
    
    def classify_heading_level(self, text, font_size, is_bold, avg_font_size):
        """Classify heading level based on various factors"""
        
        # Pattern-based classification
        level_by_pattern = self.get_level_by_pattern(text)
        if level_by_pattern:
            return level_by_pattern
        
        # Font size based classification
        font_ratio = font_size / avg_font_size if avg_font_size > 0 else 1
        
        if font_ratio >= 1.5 or (font_ratio >= 1.3 and is_bold):
            return "H1"
        elif font_ratio >= 1.2 or (font_ratio >= 1.1 and is_bold):
            return "H2"
        elif font_ratio >= 1.1 or is_bold:
            return "H3"
        
        return None
    
    def get_level_by_pattern(self, text):
        """Determine heading level based on text patterns"""
        text = text.strip()
        
        # H1 patterns
        h1_patterns = [
            r'^(\d+\.?\s+[A-Z])',  # "1. " or "1 "
            r'^(Chapter\s+\d+)',
            r'^([IVX]+\.?\s)',     # Roman numerals
            r'^([A-Z][A-Z\s]{5,})', # Long uppercase
        ]
        
        # H2 patterns  
        h2_patterns = [
            r'^(\d+\.\d+\.?\s)',   # "1.1 " or "1.1. "
            r'^(Section\s+\d+)',
            r'^([A-Z]\.\s)',       # "A. "
        ]
        
        # H3 patterns
        h3_patterns = [
            r'^(\d+\.\d+\.\d+\.?\s)', # "1.1.1 "
            r'^([a-z]\)\s)',          # "a) "
            r'^([ivx]+\.?\s)',        # lowercase roman
        ]
        
        for pattern in h1_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H1"
                
        for pattern in h2_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H2"
                
        for pattern in h3_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return "H3"
        
        return None
    
    def is_likely_heading(self, text, font_size, is_bold, avg_font_size):
        """Determine if text is likely a heading"""
        text = text.strip()
        
        # Filter out common non-headings
        if len(text) < 3 or len(text) > 200:
            return False
            
        # Skip page numbers, dates, etc.
        if re.match(r'^\d+$', text) or re.match(r'^page\s+\d+', text, re.IGNORECASE):
            return False
            
        # Skip URLs and email addresses
        if '@' in text or 'http' in text.lower() or 'www.' in text.lower():
            return False
        
        # Check patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Font-based check
        font_ratio = font_size / avg_font_size if avg_font_size > 0 else 1
        if font_ratio >= 1.2 or (font_ratio >= 1.1 and is_bold):
            return True
            
        return False
    
    def extract_headings(self, pages_data):
        """Extract headings from the document"""
        all_font_sizes = []
        
        # Calculate average font size
        for page in pages_data:
            for line in page['lines']:
                all_font_sizes.append(line['font_size'])
        
        avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12
        
        headings = []
        
        for page in pages_data:
            for line in page['lines']:
                text = line['text'].strip()
                
                if self.is_likely_heading(text, line['font_size'], line['is_bold'], avg_font_size):
                    level = self.classify_heading_level(text, line['font_size'], line['is_bold'], avg_font_size)
                    
                    if level:
                        headings.append({
                            "level": level,
                            "text": text,
                            "page": page['page_num']
                        })
        
        return self.refine_headings(headings)
    
    def refine_headings(self, headings):
        """Refine and filter headings to improve accuracy"""
        if not headings:
            return headings
        
        # Remove duplicates while preserving order
        seen = set()
        refined = []
        
        for heading in headings:
            key = (heading['text'].lower(), heading['page'])
            if key not in seen:
                seen.add(key)
                refined.append(heading)
        
        # Ensure logical hierarchy
        final_headings = []
        for i, heading in enumerate(refined):
            # Skip if this looks like a continuation of previous heading
            if i > 0:
                prev_text = refined[i-1]['text'].lower()
                curr_text = heading['text'].lower()
                
                # Skip if texts are very similar
                if self.text_similarity(prev_text, curr_text) > 0.8:
                    continue
            
            final_headings.append(heading)
        
        return final_headings
    
    def text_similarity(self, text1, text2):
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def process_pdf(self, pdf_path):
        """Process a single PDF and return structured outline"""
        try:
            logger.info(f"Processing {pdf_path}")
            
            # Extract text with formatting
            pages_data = self.extract_text_with_formatting(pdf_path)
            
            if not pages_data:
                return {"title": "Empty Document", "outline": []}
            
            # Extract title
            title = self.extract_title(pages_data)
            
            # Extract headings
            headings = self.extract_headings(pages_data)
            
            result = {
                "title": title,
                "outline": headings
            }
            
            logger.info(f"Extracted {len(headings)} headings from {pdf_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "Error Processing Document", "outline": []}

def main():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFStructureExtractor()
    
    # Process all PDF files in input directory
    for pdf_file in input_dir.glob("*.pdf"):
        logger.info(f"Found PDF: {pdf_file.name}")
        
        # Process the PDF
        result = extractor.process_pdf(str(pdf_file))
        
        # Generate output filename
        output_file = output_dir / f"{pdf_file.stem}.json"
        
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved result to {output_file}")

# if __name__ == "__main__":
#     main()
def run_cli():
    """Run the CLI version"""
    main()

if __name__ == "__main__":
    run_cli()