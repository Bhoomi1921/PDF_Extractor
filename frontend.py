import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import sys

# Import your backend classes (assuming they're in the same directory or installed)
try:
    from backend import PDFStructureExtractor, PersonaDocumentAnalyzer, IntegratedPDFProcessor
except ImportError:
    st.error("Backend modules not found. Make sure your PDF processing code is available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PDF Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(""" 
<style>     
    /* Main title header */
    .main-header {         
        font-size: 2.5rem;         
        color: #d1b3ff;  /* Light purple matching card tone */
        text-align: center;         
        margin-bottom: 2rem;
        font-weight: bold;     
    }

    /* Subheading (Round title) */
    .round-header {         
        font-size: 1.8rem;         
        color: #e0b3ff;  /* Matching tone */       
        border-left: 4px solid #e0b3ff;         
        padding-left: 1rem;         
        margin: 1.5rem 0;
        font-weight: bold;     
    }     

    /* Cards */
    .section-card, .relevance-high, .relevance-medium, .relevance-low {
        background: linear-gradient(135deg, #5e35b1 0%, #7e57c2 100%);         
        padding: 1rem;         
        border-radius: 0.5rem;         
        border-left: 3px solid #d1c4e9;         
        margin: 0.5rem 0;
        color: #f3e5f5;
        box-shadow: 0 2px 8px rgba(0,1,5,0.3);
    }     
</style> 
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processor' not in st.session_state:
        st.session_state.processor = IntegratedPDFProcessor()
    if 'round1a_results' not in st.session_state:
        st.session_state.round1a_results = {}
    if 'round1b_results' not in st.session_state:
        st.session_state.round1b_results = {}

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files to temporary directory and return paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths

def display_outline_tree(outline: List[Dict[str, Any]]):
    """Display document outline in a tree-like structure"""
    if not outline:
        st.info("No headings detected in the document.")
        return
    
    for item in outline:
        level = item.get('level', 'H2')
        text = item.get('text', '')
        page = item.get('page', 1)
        
        # Create indentation based on heading level
        indent = "    " * (int(level[1]) - 1) if level.startswith('H') else ""
        
        # Different styling for different levels
        if level == 'H1':
            st.markdown(f"{indent}**🔸 {text}** *(Page {page})*")
        elif level == 'H2':
            st.markdown(f"{indent}• {text} *(Page {page})*")
        else:  # H3
            st.markdown(f"{indent}  ◦ {text} *(Page {page})*")

def display_relevance_badge(score: float) -> str:
    """Generate relevance badge based on score"""
    if score >= 0.7:
        return "🟢 High"
    elif score >= 0.4:
        return "🟡 Medium"
    else:
        return "🔴 Low"

def round_1a_interface():
    """Round 1A: PDF Structure Extraction Interface"""
    st.markdown('<div class="round-header">Round 1A: PDF Structure Extraction</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Goal:** Extract structured outlines (Title, H1, H2, H3 headings) from PDFs with page numbers.
    Perfect for understanding document hierarchy and navigation.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload PDF files for structure extraction",
        type=['pdf'],
        accept_multiple_files=True,
        key="round1a_files"
    )
    
    if uploaded_files:
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 Extract Structures", key="extract_btn"):
                with st.spinner("Extracting document structures..."):
                    file_paths = save_uploaded_files(uploaded_files)
                    results = {}
                    
                    progress_bar = st.progress(0)
                    for i, file_path in enumerate(file_paths):
                        try:
                            result = st.session_state.processor.process_single_pdf(file_path)
                            filename = Path(file_path).name
                            results[filename] = result
                            progress_bar.progress((i + 1) / len(file_paths))
                        except Exception as e:
                            st.error(f"Error processing {Path(file_path).name}: {str(e)}")
                    
                    st.session_state.round1a_results = results
        
        # Display results
        if st.session_state.round1a_results:
            st.markdown("### 📊 Extraction Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            total_docs = len(st.session_state.round1a_results)
            total_headings = sum(len(r.get('outline', [])) for r in st.session_state.round1a_results.values())
            
            with col1:
                st.metric("Documents Processed", total_docs)
            with col2:
                st.metric("Total Headings", total_headings)
            with col3:
                avg_headings = total_headings / total_docs if total_docs > 0 else 0
                st.metric("Avg Headings/Doc", f"{avg_headings:.1f}")
            
            # Individual document results
            for filename, result in st.session_state.round1a_results.items():
                with st.expander(f"📄 {filename} - {result.get('title', 'Unknown Title')}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Document Outline:**")
                        display_outline_tree(result.get('outline', []))
                    
                    with col2:
                        st.markdown("**Metadata:**")
                        st.write(f"**Title:** {result.get('title', 'N/A')}")
                        st.write(f"**Headings Found:** {len(result.get('outline', []))}")
                        
                        # Heading level breakdown
                        outline = result.get('outline', [])
                        level_counts = {}
                        for item in outline:
                            level = item.get('level', 'Unknown')
                            level_counts[level] = level_counts.get(level, 0) + 1
                        
                        if level_counts:
                            st.markdown("**Level Breakdown:**")
                            for level, count in sorted(level_counts.items()):
                                st.write(f"{level}: {count}")
                        
                        # Download JSON result
                        json_str = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_str,
                            file_name=f"{Path(filename).stem}_structure.json",
                            mime="application/json"
                        )

def round_1b_interface():
    """Round 1B: Persona-Driven Document Intelligence Interface"""
    st.markdown('<div class="round-header">Round 1B: Persona-Driven Document Intelligence</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Goal:** Extract and rank the most relevant sections from multiple documents based on a specific persona and their job-to-be-done.
    Ideal for targeted research and analysis.
    """)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF documents for analysis (3-10 recommended)",
        type=['pdf'],
        accept_multiple_files=True,
        key="round1b_files"
    )
    
    # Persona and Job Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Persona Definition")
        
        # Preset personas for quick selection
        preset_personas = {
            "Custom": "",
            "PhD Researcher": "PhD Researcher in Computational Biology with expertise in machine learning applications for drug discovery",
            "Investment Analyst": "Investment Analyst specializing in technology sector analysis and financial modeling",
            "Undergraduate Student": "Undergraduate Chemistry Student preparing for advanced organic chemistry examinations",
            "Business Consultant": "Business Consultant focusing on strategic analysis and competitive intelligence",
            "Data Scientist": "Data Scientist working on natural language processing and information extraction projects"
        }
        
        selected_persona = st.selectbox("Choose a preset persona or select Custom:", list(preset_personas.keys()))
        
        if selected_persona == "Custom":
            persona = st.text_area(
                "Describe the persona (role, expertise, focus areas):",
                placeholder="e.g., PhD Researcher in Computational Biology with expertise in...",
                height=100
            )
        else:
            persona = preset_personas[selected_persona]
            st.text_area("Persona Description:", value=persona, height=100, disabled=True)
    
    with col2:
        st.markdown("### 🎯 Job-to-be-Done")
        
        # Preset jobs for quick selection
        preset_jobs = {
            "Custom": "",
            "Literature Review": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
            "Financial Analysis": "Analyze revenue trends, R&D investments, and market positioning strategies",
            "Exam Preparation": "Identify key concepts and mechanisms for exam preparation on reaction kinetics",
            "Market Research": "Conduct competitive analysis and identify market opportunities and threats",
            "Technical Summary": "Extract technical specifications and implementation details for system architecture"
        }
        
        selected_job = st.selectbox("Choose a preset job or select Custom:", list(preset_jobs.keys()))
        
        if selected_job == "Custom":
            job_to_be_done = st.text_area(
                "Describe the specific task/goal:",
                placeholder="e.g., Prepare a comprehensive literature review focusing on...",
                height=100
            )
        else:
            job_to_be_done = preset_jobs[selected_job]
            st.text_area("Job Description:", value=job_to_be_done, height=100, disabled=True)
    
    # Analysis button
    if uploaded_files and persona and job_to_be_done:
        if st.button("🧠 Analyze Documents", key="analyze_btn"):
            with st.spinner("Analyzing documents with persona intelligence..."):
                try:
                    file_paths = save_uploaded_files(uploaded_files)
                    
                    result = st.session_state.processor.process_document_collection(
                        file_paths, persona, job_to_be_done
                    )
                    
                    st.session_state.round1b_results = result
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    # Display results
    if st.session_state.round1b_results:
        result = st.session_state.round1b_results
        metadata = result.get('metadata', {})
        
        st.markdown("### 📈 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents Analyzed", metadata.get('total_documents', 0))
        with col2:
            st.metric("Sections Found", metadata.get('total_sections_found', 0))
        with col3:
            st.metric("Top Sections", len(result.get('extracted_sections', [])))
        with col4:
            st.metric("Sub-sections", len(result.get('sub_section_analysis', [])))
        
        # Document structures
        if result.get('document_structures'):
            st.markdown("### 📚 Document Structures")
            for doc in result['document_structures']:
                with st.expander(f"📄 {doc['filename']} - {doc['title']}"):
                    display_outline_tree(doc.get('outline', []))
        
        # Top relevant sections
        if result.get('extracted_sections'):
            st.markdown("### 🎯 Most Relevant Sections")
            
            for section in result['extracted_sections'][:10]:  # Top 10 sections
                relevance_score = section.get('relevance_score', 0)
                relevance_badge = display_relevance_badge(relevance_score)
                
                css_class = "relevance-high" if relevance_score >= 0.7 else "relevance-medium" if relevance_score >= 0.4 else "relevance-low"
                
                st.markdown(f'''
                <div class="section-card {css_class}">
                    <strong>#{section.get('importance_rank', 'N/A')} - {section.get('section_title', 'Unknown Section')}</strong><br>
                    <small>📄 {section.get('document', 'Unknown')} | 📄 Page {section.get('page_number', 'N/A')} | {relevance_badge} ({relevance_score:.3f})</small>
                </div>
                ''', unsafe_allow_html=True)
        
        # Sub-section analysis
        if result.get('sub_section_analysis'):
            st.markdown("### 🔍 Detailed Sub-section Analysis")
            
            for i, sub_section in enumerate(result['sub_section_analysis'][:8]):  # Top 8 sub-sections
                relevance_score = sub_section.get('relevance_score', 0)
                relevance_badge = display_relevance_badge(relevance_score)
                
                with st.expander(f"Sub-section {i+1} - {relevance_badge} ({relevance_score:.3f})"):
                    st.markdown(f"**Document:** {sub_section.get('document', 'Unknown')}")
                    st.markdown(f"**Page:** {sub_section.get('page_number', 'N/A')}")
                    st.markdown("**Content:**")
                    st.markdown(sub_section.get('refined_text', 'No content available'))
        
        # Download results
        col1, col2 = st.columns(2)
        with col1:
            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Full Analysis (JSON)",
                data=json_str,
                file_name=f"persona_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Create summary report
            summary_report = f"""# Persona-Driven Document Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Configuration
- **Persona:** {metadata.get('persona', 'N/A')}
- **Job-to-be-Done:** {metadata.get('job_to_be_done', 'N/A')}

## Summary
- Documents Analyzed: {metadata.get('total_documents', 0)}
- Total Sections: {metadata.get('total_sections_found', 0)}
- Top Relevant Sections: {len(result.get('extracted_sections', []))}

## Top Sections
"""
            
            for section in result.get('extracted_sections', [])[:5]:
                summary_report += f"\n{section.get('importance_rank', 'N/A')}. **{section.get('section_title', 'Unknown')}** (Score: {section.get('relevance_score', 0):.3f})\n   - Document: {section.get('document', 'Unknown')}\n   - Page: {section.get('page_number', 'N/A')}\n"
            
            st.download_button(
                label="📋 Download Summary Report (MD)",
                data=summary_report,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📄 PDF Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("**Connecting the Dots Challenge** - Adobe India Hackathon 2025")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    selected_round = st.sidebar.selectbox(
        "Choose Round:",
        ["Round 1A: Structure Extraction", "Round 1B: Persona Intelligence", "About"]
    )
    
    # Main content area
    if selected_round == "Round 1A: Structure Extraction":
        round_1a_interface()
    
    elif selected_round == "Round 1B: Persona Intelligence":
        round_1b_interface()
    
    elif selected_round == "About":
        st.markdown("## 🎯 About This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Round 1A: Structure Extraction
            - **Purpose:** Extract hierarchical document structure
            - **Input:** Single or multiple PDF files
            - **Output:** Title and heading outline (H1, H2, H3)
            - **Use Cases:** 
              - Document navigation
              - Table of contents generation
              - Structure analysis
            """)
        
        with col2:
            st.markdown("""
            ### Round 1B: Persona Intelligence
            - **Purpose:** Context-aware document analysis
            - **Input:** Document collection + persona + job-to-be-done
            - **Output:** Ranked relevant sections and sub-sections
            - **Use Cases:**
              - Targeted research
              - Literature reviews
              - Competitive analysis
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🚀 How to Use
        1. **Choose your round** from the sidebar
        2. **Upload PDF documents** using the file uploader
        3. **Configure parameters** (persona & job for Round 1B)
        4. **Run analysis** and explore results
        5. **Download outputs** in JSON or Markdown format
        
        ### 💡 Tips
        - For Round 1A: Works best with structured documents (reports, papers, books)
        - For Round 1B: Provide detailed persona descriptions for better results
        - Use 3-10 documents for optimal Round 1B performance
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built with:**")
    st.sidebar.markdown("• Streamlit")
    st.sidebar.markdown("• PyMuPDF")
    st.sidebar.markdown("• scikit-learn")
    st.sidebar.markdown("• NLTK")

if __name__ == "__main__":
    main()