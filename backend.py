import os
import json
import re
from pathlib import Path
import fitz  # PyMuPDF
from collections import Counter, defaultdict
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFStructureExtractor:
    """Round 1A: PDF Structure Extraction"""
    
    def __init__(self):
        self.heading_patterns = [
            r'^(\d+\.?\s+[A-Z])',  # "1. Introduction"
            r'^([A-Z][A-Z\s]{3,})',  # "INTRODUCTION"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Title Case"
            r'^(Chapter\s+\d+)',
            r'^(Section\s+\d+)',
            r'^([IVX]+\.?\s+[A-Z])',  # Roman numerals
            r'^([A-Z]\.\s+[A-Z])',  # "A. Section"
        ]
    
    def is_likely_heading(self, text: str, font_size: float = None, 
                         avg_font_size: float = None) -> bool:
        """Determine if text is likely a heading"""
        if not text or len(text.strip()) < 3 or len(text.strip()) > 200:
            return False
        
        text = text.strip()
        
        # Pattern matching
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Font size analysis (if available)
        if font_size and avg_font_size and font_size > avg_font_size * 1.1:
            return True
        
        # Structural characteristics
        words = text.split()
        if len(words) <= 8 and not text.endswith('.') and not text.endswith(','):
            # Check capitalization
            capitalized_count = sum(1 for word in words if word and word[0].isupper())
            if capitalized_count >= len(words) * 0.6:
                return True
        
        return False
    
    def classify_heading_level(self, text: str, font_size: float = None) -> str:
        """Classify heading level (H1, H2, H3)"""
        text = text.strip()
        
        # H1 patterns (major sections)
        h1_patterns = [
            r'^(Chapter\s+\d+)',
            r'^([IVX]+\.?\s)',
            r'^(\d+\.\s+[A-Z])',
            r'^([A-Z][A-Z\s]{5,})',  # Long all-caps
        ]
        
        for pattern in h1_patterns:
            if re.match(pattern, text):
                return 'H1'
        
        # H3 patterns (sub-sections)
        h3_patterns = [
            r'^(\d+\.\d+\.\d+)',
            r'^([a-z]\)\s)',
            r'^([a-z]\.\s)',
        ]
        
        for pattern in h3_patterns:
            if re.match(pattern, text):
                return 'H3'
        
        # H2 by default (intermediate level)
        return 'H2'
    
    def extract_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structure from PDF (Round 1A functionality)"""
        try:
            doc = fitz.open(pdf_path)
            outline = []
            title = "Unknown Title"
            
            # Try to get title from metadata
            metadata = doc.metadata
            if metadata and metadata.get('title'):
                title = metadata['title']
            
            # Extract headings from content
            font_sizes = []
            all_text_blocks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with font information
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                
                                if text and len(text) > 3:
                                    font_sizes.append(font_size)
                                    all_text_blocks.append({
                                        'text': text,
                                        'font_size': font_size,
                                        'page': page_num + 1
                                    })
            
            # Calculate average font size
            avg_font_size = np.mean(font_sizes) if font_sizes else 12
            
            # Extract headings
            for block in all_text_blocks:
                text = block['text']
                font_size = block['font_size']
                page = block['page']
                
                if self.is_likely_heading(text, font_size, avg_font_size):
                    level = self.classify_heading_level(text, font_size)
                    
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": page
                    })
            
            # If no title found, use first H1 or filename
            if title == "Unknown Title" and outline:
                h1_headings = [h for h in outline if h['level'] == 'H1']
                if h1_headings:
                    title = h1_headings[0]['text']
                else:
                    title = Path(pdf_path).stem
            
            doc.close()
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            logger.error(f"Error extracting structure from {pdf_path}: {str(e)}")
            return {"title": Path(pdf_path).name, "outline": []}
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Main processing function for Round 1A"""
        return self.extract_pdf_structure(pdf_path)

class PersonaDocumentAnalyzer:
    """Round 1B: Persona-Driven Document Intelligence"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        # Initialize structure extractor for enhanced analysis
        self.structure_extractor = PDFStructureExtractor()
        
    def extract_full_document_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract full content with sections from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            # Get structure from Round 1A
            structure_result = self.structure_extractor.extract_pdf_structure(pdf_path)
            
            content = {
                'filename': Path(pdf_path).name,
                'title': structure_result.get('title', 'Unknown Title'),
                'pages': [],
                'sections': [],
                'outline': structure_result.get('outline', [])
            }
            
            current_section = None
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                content['pages'].append({
                    'page_num': page_num + 1,
                    'text': page_text
                })
                
                # Use extracted outline to identify sections more accurately
                page_headings = [h for h in content['outline'] if h['page'] == page_num + 1]
                
                if page_headings:
                    # Process sections based on extracted headings
                    lines = page_text.split('\n')
                    current_content = ""
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Check if this line matches any heading
                        is_heading = any(heading['text'].strip() == line for heading in page_headings)
                        
                        if is_heading:
                            # Save previous section
                            if current_section:
                                current_section['content'] = current_content.strip()
                                content['sections'].append(current_section)
                            
                            # Start new section
                            heading_info = next(h for h in page_headings if h['text'].strip() == line)
                            current_section = {
                                'title': line,
                                'page': page_num + 1,
                                'level': heading_info['level'],
                                'content': ''
                            }
                            current_content = ""
                        elif current_section and line:
                            current_content += line + ' '
                    
                    # Add remaining content to current section
                    if current_section and current_content:
                        current_section['content'] = current_content.strip()
                else:
                    # Fallback to basic section detection
                    lines = page_text.split('\n')
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if self.is_section_heading(line):
                            if current_section:
                                content['sections'].append(current_section)
                            
                            current_section = {
                                'title': line,
                                'page': page_num + 1,
                                'level': 'H2',  # Default level
                                'content': '',
                                'start_line': line_num
                            }
                        elif current_section and line:
                            current_section['content'] += line + ' '
            
            # Add last section
            if current_section:
                content['sections'].append(current_section)
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {str(e)}")
            return {'filename': Path(pdf_path).name, 'pages': [], 'sections': []}
    
    def is_section_heading(self, line: str) -> bool:
        """Determine if a line is a section heading (fallback method)"""
        if len(line) < 3 or len(line) > 200:
            return False
            
        # Use the same logic as PDFStructureExtractor
        return self.structure_extractor.is_likely_heading(line)
    
    def create_persona_keywords(self, persona: str, job_to_be_done: str) -> List[str]:
        """Extract relevant keywords from persona and job description"""
        combined_text = f"{persona} {job_to_be_done}".lower()
        
        words = word_tokenize(combined_text)
        keywords = []
        
        for word in words:
            if (len(word) > 3 and 
                word not in self.stop_words and 
                word.isalpha() and
                not word in ['person', 'people', 'work', 'need', 'want']):
                keywords.append(self.stemmer.stem(word))
        
        # Add domain-specific terms
        domain_keywords = self.get_domain_keywords(persona, job_to_be_done)
        keywords.extend(domain_keywords)
        
        return list(set(keywords))
    
    def get_domain_keywords(self, persona: str, job: str) -> List[str]:
        """Get domain-specific keywords based on persona"""
        domain_map = {
            'researcher': ['research', 'study', 'analysis', 'method', 'result', 'conclusion', 'hypothesis', 'data', 'experiment'],
            'student': ['concept', 'theory', 'principle', 'example', 'definition', 'explanation', 'practice', 'exercise'],
            'analyst': ['trend', 'pattern', 'metric', 'performance', 'comparison', 'evaluation', 'assessment', 'report'],
            'investment': ['revenue', 'profit', 'growth', 'market', 'financial', 'strategy', 'competition', 'risk'],
            'phd': ['methodology', 'literature', 'review', 'framework', 'approach', 'validation', 'benchmark'],
            'business': ['strategy', 'market', 'customer', 'revenue', 'growth', 'competitive', 'opportunity'],
            'chemistry': ['reaction', 'mechanism', 'synthesis', 'compound', 'molecular', 'kinetic', 'catalyst'],
            'biology': ['cellular', 'molecular', 'genetic', 'protein', 'enzyme', 'pathway', 'organism'],
            'financial': ['balance', 'sheet', 'income', 'cash', 'flow', 'asset', 'liability', 'equity']
        }
        
        keywords = []
        text = f"{persona} {job}".lower()
        
        for domain, terms in domain_map.items():
            if domain in text:
                keywords.extend(terms)
        
        return keywords
    
    def calculate_section_relevance(self, section: Dict[str, Any], persona_keywords: List[str], 
                                  job_keywords: List[str]) -> float:
        """Calculate relevance score for a section"""
        section_text = f"{section.get('title', '')} {section.get('content', '')}".lower()
        
        if not section_text.strip():
            return 0.0
        
        # Keyword matching score
        keyword_score = 0
        all_keywords = persona_keywords + job_keywords
        
        for keyword in all_keywords:
            if keyword in section_text:
                # Title matches get higher weight
                if keyword in section.get('title', '').lower():
                    keyword_score += 2
                else:
                    keyword_score += 1
        
        # Normalize by section length and keyword count
        text_length = len(section_text.split())
        keyword_density = keyword_score / max(text_length, 1) * 100
        
        # Content quality indicators
        quality_score = 0
        
        # Prefer sections with substantial content
        if text_length > 50:
            quality_score += 0.2
        
        # Prefer sections with numbers/statistics for analytical personas
        if re.search(r'\d+\.?\d*%|\$\d+|\d+\.\d+', section_text):
            quality_score += 0.1
        
        # Prefer sections with technical terms
        technical_indicators = ['method', 'approach', 'analysis', 'result', 'conclusion', 'data']
        for indicator in technical_indicators:
            if indicator in section_text:
                quality_score += 0.05
        
        # Boost score based on heading level (H1 > H2 > H3)
        level_boost = {'H1': 0.3, 'H2': 0.2, 'H3': 0.1}.get(section.get('level', 'H2'), 0.1)
        quality_score += level_boost
        
        final_score = keyword_density + quality_score
        return min(1.0, final_score / 10)  # Normalize to 0-1
    
    def extract_sub_sections(self, section: Dict[str, Any], persona_keywords: List[str], 
                           job_keywords: List[str]) -> List[Dict[str, Any]]:
        """Extract and rank sub-sections from a section"""
        content = section.get('content', '')
        if not content:
            return []
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # If no paragraph breaks, split by sentences
        if len(paragraphs) <= 2:
            sentences = sent_tokenize(content)
            paragraphs = []
            for i in range(0, len(sentences), 3):
                chunk = ' '.join(sentences[i:i+3])
                if chunk.strip():
                    paragraphs.append(chunk.strip())
        
        sub_sections = []
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) >= 10:  # Minimum meaningful content
                relevance = self.calculate_text_relevance(paragraph, persona_keywords, job_keywords)
                
                sub_sections.append({
                    'document': section.get('document', ''),
                    'refined_text': paragraph[:500] + '...' if len(paragraph) > 500 else paragraph,
                    'page_number': section.get('page', 1),
                    'relevance_score': relevance
                })
        
        # Sort by relevance and return top sub-sections
        sub_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        return sub_sections[:5]  # Top 5 sub-sections
    
    def calculate_text_relevance(self, text: str, persona_keywords: List[str], 
                               job_keywords: List[str]) -> float:
        """Calculate relevance score for a piece of text"""
        text_lower = text.lower()
        
        # Keyword matching
        keyword_matches = 0
        all_keywords = persona_keywords + job_keywords
        
        for keyword in all_keywords:
            if keyword in text_lower:
                keyword_matches += text_lower.count(keyword)
        
        # Text characteristics
        word_count = len(text.split())
        keyword_density = keyword_matches / max(word_count, 1)
        
        # Quality indicators
        quality_score = 0
        if re.search(r'\d+', text):  # Contains numbers
            quality_score += 0.1
        if re.search(r'[.!?]', text):  # Complete sentences
            quality_score += 0.1
        if len(text.split()) > 20:  # Substantial content
            quality_score += 0.1
        
        return min(1.0, keyword_density * 10 + quality_score)
    
    def rank_sections(self, all_sections: List[Dict[str, Any]], persona: str, 
                     job_to_be_done: str, max_sections: int = 10) -> List[Dict[str, Any]]:
        """Rank sections based on persona and job relevance"""
        persona_keywords = self.create_persona_keywords(persona, '')
        job_keywords = self.create_persona_keywords('', job_to_be_done)
        
        # Calculate relevance for each section
        for section in all_sections:
            relevance = self.calculate_section_relevance(section, persona_keywords, job_keywords)
            section['relevance_score'] = relevance
        
        # Sort by relevance
        ranked_sections = sorted(all_sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Return top sections with importance rank
        result = []
        for i, section in enumerate(ranked_sections[:max_sections]):
            section['importance_rank'] = i + 1
            result.append(section)
        
        return result
    
    def process_document_collection(self, pdf_paths: List[str], persona: str, 
                                  job_to_be_done: str) -> Dict[str, Any]:
        """Process collection of documents for persona-driven analysis"""
        try:
            logger.info(f"Processing {len(pdf_paths)} documents for persona analysis")
            
            # Extract content from all documents
            all_documents = []
            all_sections = []
            
            for pdf_path in pdf_paths:
                doc_content = self.extract_full_document_content(pdf_path)
                all_documents.append(doc_content)
                
                # Add document reference to sections
                for section in doc_content['sections']:
                    section['document'] = doc_content['filename']
                    all_sections.append(section)
            
            # Rank sections based on persona and job
            ranked_sections = self.rank_sections(all_sections, persona, job_to_be_done)
            
            # Extract sub-sections for top sections
            persona_keywords = self.create_persona_keywords(persona, '')
            job_keywords = self.create_persona_keywords('', job_to_be_done)
            
            all_sub_sections = []
            for section in ranked_sections[:5]:  # Top 5 sections
                sub_sections = self.extract_sub_sections(section, persona_keywords, job_keywords)
                all_sub_sections.extend(sub_sections)
            
            # Sort all sub-sections by relevance
            all_sub_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Prepare output
            result = {
                "metadata": {
                    "input_documents": [
                        {
                            "filename": doc['filename'],
                            "title": doc['title'],
                            "sections_count": len([s for s in all_sections if s['document'] == doc['filename']])
                        }
                        for doc in all_documents
                    ],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "total_sections_found": len(all_sections),
                    "total_documents": len(pdf_paths)
                },
                "document_structures": [
                    {
                        "filename": doc['filename'],
                        "title": doc['title'],
                        "outline": doc['outline']
                    }
                    for doc in all_documents
                ],
                "extracted_sections": [
                    {
                        "document": section['document'],
                        "page_number": section['page'],
                        "section_title": section['title'],
                        "section_level": section.get('level', 'H2'),
                        "importance_rank": section['importance_rank'],
                        "relevance_score": round(section['relevance_score'], 3)
                    }
                    for section in ranked_sections
                ],
                "sub_section_analysis": [
                    {
                        "document": sub_section['document'],
                        "refined_text": sub_section['refined_text'],
                        "page_number": sub_section['page_number'],
                        "relevance_score": round(sub_section['relevance_score'], 3)
                    }
                    for sub_section in all_sub_sections[:15]  # Top 15 sub-sections
                ]
            }
            
            logger.info(f"Analysis complete: {len(ranked_sections)} sections, {len(all_sub_sections)} sub-sections")
            return result
            
        except Exception as e:
            logger.error(f"Error in document collection analysis: {str(e)}")
            return {
                "metadata": {
                    "input_documents": [Path(p).name for p in pdf_paths],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                },
                "document_structures": [],
                "extracted_sections": [],
                "sub_section_analysis": []
            }

class IntegratedPDFProcessor:
    """Main class that integrates Round 1A and 1B functionality"""
    
    def __init__(self):
        self.structure_extractor = PDFStructureExtractor()
        self.persona_analyzer = PersonaDocumentAnalyzer()
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process single PDF for structure extraction (Round 1A)"""
        return self.structure_extractor.process_pdf(pdf_path)
    
    def process_document_collection(self, pdf_paths: List[str], persona: str, 
                                  job_to_be_done: str) -> Dict[str, Any]:
        """Process document collection for persona analysis (Round 1B)"""
        return self.persona_analyzer.process_document_collection(pdf_paths, persona, job_to_be_done)
    
    def batch_process_pdfs(self, input_dir: str, output_dir: str) -> None:
        """Batch process PDFs for Round 1A (Docker execution)"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_path.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                result = self.process_single_pdf(str(pdf_file))
                output_file = output_path / f"{pdf_file.stem}.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processed {pdf_file.name} -> {output_file.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")

def main():
    """CLI entry point"""
    processor = IntegratedPDFProcessor()
    
    # Check if we're running in Docker mode (Round 1A)
    if os.path.exists("/app/input") and os.path.exists("/app/output"):
        # Check if persona config exists (Round 1B mode)
        config_file = Path("/app/input/persona_config.json")
        
        if config_file.exists():
            # Round 1B mode - Persona analysis
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                persona = config.get('persona', '')
                job_to_be_done = config.get('job_to_be_done', '')
                
                if not persona or not job_to_be_done:
                    logger.error("persona and job_to_be_done must be specified in config")
                    return
                
                # Find all PDF files
                pdf_files = list(Path("/app/input").glob("*.pdf"))
                if not pdf_files:
                    logger.error("No PDF files found in input directory")
                    return
                
                logger.info(f"Running Round 1B: Processing {len(pdf_files)} files with persona analysis")
                
                # Process documents
                result = processor.process_document_collection(
                    [str(pdf) for pdf in pdf_files],
                    persona,
                    job_to_be_done
                )
                
                # Save result
                output_file = Path("/app/output") / "persona_analysis_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Round 1B results saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error in Round 1B execution: {str(e)}")
        else:
            # Round 1A mode - Structure extraction
            logger.info("Running Round 1A: PDF structure extraction")
            processor.batch_process_pdfs("/app/input", "/app/output")
    else:
        logger.info("Not running in Docker mode. Use the integrated frontend for testing.")

if __name__ == "__main__":
    main()