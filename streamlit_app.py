import streamlit as st
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time

# Import your PDF extractor class
from extract_pdf_structure import PDFStructureExtractor  # Your original code should be in pdf_extractor.py

# Page configuration
st.set_page_config(
    page_title="PDF Structure Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .heading-level-h1 {
        color: #2c3e50;
        font-weight: bold;
        font-size: 1.2em;
        margin-left: 0px;
    }
    .heading-level-h2 {
        color: #34495e;
        font-weight: 600;
        font-size: 1.1em;
        margin-left: 20px;
    }
    .heading-level-h3 {
        color: #7f8c8d;
        font-weight: 500;
        font-size: 1.0em;
        margin-left: 40px;
    }
    .validation-correct {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
    .validation-incorrect {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitPDFValidator:
    def __init__(self):
        self.extractor = PDFStructureExtractor()
        
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>📄 PDF Structure Extractor & Validator</h1>
            <p>Upload PDFs and validate the extracted structure</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("⚙️ Settings")
        
        # Extraction settings
        st.sidebar.subheader("Extraction Settings")
        min_heading_length = st.sidebar.slider("Min Heading Length", 3, 20, 3)
        max_heading_length = st.sidebar.slider("Max Heading Length", 50, 300, 200)
        font_size_threshold = st.sidebar.slider("Font Size Threshold", 1.0, 2.0, 1.2, 0.1)
        
        # Validation settings
        st.sidebar.subheader("Validation Settings")
        show_confidence = st.sidebar.checkbox("Show Confidence Scores", True)
        highlight_issues = st.sidebar.checkbox("Highlight Potential Issues", True)
        
        return {
            'min_heading_length': min_heading_length,
            'max_heading_length': max_heading_length,
            'font_size_threshold': font_size_threshold,
            'show_confidence': show_confidence,
            'highlight_issues': highlight_issues
        }
    
    def process_uploaded_file(self, uploaded_file, settings):
        """Process uploaded PDF file"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Process with extractor
            with st.spinner("🔍 Analyzing PDF structure..."):
                result = self.extractor.process_pdf(tmp_file_path)
            
            # Clean up
            os.unlink(tmp_file_path)
            
            return result
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            return None
    
    def calculate_confidence_score(self, heading):
        """Calculate confidence score for a heading"""
        score = 0.5  # Base score
        
        text = heading['text'].strip()
        
        # Length-based scoring
        if 5 <= len(text) <= 100:
            score += 0.2
        elif len(text) > 100:
            score -= 0.1
            
        # Pattern-based scoring
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered headings
            r'^Chapter\s+\d+',    # Chapter headings
            r'^[IVX]+\.?\s',      # Roman numerals
            r'^[A-Z][A-Z\s]{2,}', # All caps
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                score += 0.3
                break
        
        # Level consistency
        if heading['level'] in ['H1', 'H2', 'H3']:
            score += 0.2
            
        return min(1.0, max(0.0, score))
    
    def identify_potential_issues(self, headings):
        """Identify potential issues in extracted headings"""
        issues = []
        
        if not headings:
            return ["No headings found - document might be poorly structured"]
        
        # Check for too many H1s
        h1_count = sum(1 for h in headings if h['level'] == 'H1')
        if h1_count > len(headings) * 0.3:  # More than 30% are H1s
            issues.append(f"Too many H1 headings ({h1_count}) - might indicate over-classification")
        
        # Check for very short headings
        short_headings = [h for h in headings if len(h['text'].strip()) < 5]
        if len(short_headings) > len(headings) * 0.2:
            issues.append(f"{len(short_headings)} very short headings detected - might be false positives")
        
        # Check for missing hierarchy
        levels = [h['level'] for h in headings]
        if 'H1' in levels and 'H3' in levels and 'H2' not in levels:
            issues.append("Missing H2 level - hierarchy might be incomplete")
        
        # Check for duplicate headings
        texts = [h['text'].lower().strip() for h in headings]
        duplicates = [text for text, count in Counter(texts).items() if count > 1]
        if duplicates:
            issues.append(f"{len(duplicates)} duplicate headings found")
        
        return issues
    
    def render_extraction_results(self, result, settings):
        """Render extraction results with validation"""
        if not result or 'outline' not in result:
            st.error("❌ No results to display")
            return
        
        headings = result['outline']
        title = result.get('title', 'Unknown Title')
        
        # Display title
        st.subheader("📖 Document Title")
        st.info(f"**{title}**")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Headings", len(headings))
        
        with col2:
            h1_count = sum(1 for h in headings if h['level'] == 'H1')
            st.metric("H1 Headings", h1_count)
        
        with col3:
            h2_count = sum(1 for h in headings if h['level'] == 'H2')
            st.metric("H2 Headings", h2_count)
        
        with col4:
            h3_count = sum(1 for h in headings if h['level'] == 'H3')
            st.metric("H3 Headings", h3_count)
        
        # Validation section
        st.subheader("🔍 Validation Results")
        
        # Identify issues
        issues = self.identify_potential_issues(headings)
        
        if issues:
            st.warning("⚠️ Potential Issues Detected:")
            for issue in issues:
                st.markdown(f"• {issue}")
        else:
            st.success("✅ No obvious issues detected in the extraction")
        
        # Display headings with validation
        st.subheader("📋 Extracted Structure")
        
        if not headings:
            st.info("No headings were extracted from this document.")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📖 Structure View", "📊 Analysis", "🔧 Manual Validation"])
        
        with tab1:
            self.render_structure_view(headings, settings)
        
        with tab2:
            self.render_analysis_view(headings, settings)
        
        with tab3:
            self.render_manual_validation(headings, settings)
    
    def render_structure_view(self, headings, settings):
        """Render the structure view of headings"""
        for i, heading in enumerate(headings):
            level = heading['level']
            text = heading['text']
            page = heading.get('page', 'Unknown')
            
            # Calculate confidence if enabled
            confidence = None
            if settings['show_confidence']:
                confidence = self.calculate_confidence_score(heading)
            
            # Create heading display
            level_class = f"heading-level-{level.lower()}"
            confidence_badge = ""
            
            if confidence is not None:
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
                confidence_badge = f"<span style='color: {color}; font-size: 0.8em;'>({confidence:.2f})</span>"
            
            page_info = f"<span style='color: #888; font-size: 0.8em;'>[Page {page}]</span>"
            
            st.markdown(f"""
            <div class="{level_class}">
                {level}: {text} {confidence_badge} {page_info}
            </div>
            """, unsafe_allow_html=True)
    
    def render_analysis_view(self, headings, settings):
        """Render analysis charts and statistics"""
        if not headings:
            st.info("No data to analyze")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Heading level distribution
            level_counts = Counter([h['level'] for h in headings])
            fig_pie = px.pie(
                values=list(level_counts.values()),
                names=list(level_counts.keys()),
                title="Heading Level Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Page distribution
            page_counts = Counter([h.get('page', 'Unknown') for h in headings])
            fig_bar = px.bar(
                x=list(page_counts.keys()),
                y=list(page_counts.values()),
                title="Headings per Page"
            )
            # fig_bar.update_xaxis(title_t="Page Number")
            fig_bar.update_xaxes(title_text="Page Number")
            fig_bar.update_yaxes(title_text="Number of Headings")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence distribution if enabled
        if settings['show_confidence']:
            confidences = [self.calculate_confidence_score(h) for h in headings]
            fig_hist = px.histogram(
                x=confidences,
                nbins=10,
                title="Confidence Score Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Statistics table
        st.subheader("📈 Statistics")
        stats_data = {
            "Metric": ["Total Headings", "Unique Headings", "Average Text Length", "Pages with Headings"],
            "Value": [
                len(headings),
                len(set(h['text'] for h in headings)),
                f"{sum(len(h['text']) for h in headings) / len(headings):.1f}" if headings else "0",
                len(set(h.get('page', 0) for h in headings))
            ]
        }
        st.table(pd.DataFrame(stats_data))
    
    def render_manual_validation(self, headings, settings):
        """Render manual validation interface"""
        st.info("👤 Manually validate the extracted headings below. Mark each heading as correct or incorrect:")
        
        validation_results = {}
        
        for i, heading in enumerate(headings):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
            
            with col1:
                st.write(f"**{heading['level']}**: {heading['text']}")
            
            with col2:
                correct = st.checkbox("✅", key=f"correct_{i}", help="Mark as correctly identified")
            
            with col3:
                incorrect = st.checkbox("❌", key=f"incorrect_{i}", help="Mark as incorrectly identified")
            
            with col4:
                st.write(f"Page {heading.get('page', 'Unknown')}")
            
            # Store validation result
            if correct and not incorrect:
                validation_results[i] = 'correct'
            elif incorrect and not correct:
                validation_results[i] = 'incorrect'
            else:
                validation_results[i] = 'unknown'
        
        # Summary of validation
        if validation_results:
            correct_count = sum(1 for v in validation_results.values() if v == 'correct')
            incorrect_count = sum(1 for v in validation_results.values() if v == 'incorrect')
            total_validated = correct_count + incorrect_count
            
            if total_validated > 0:
                accuracy = (correct_count / total_validated) * 100
                
                st.subheader("🎯 Validation Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correct", correct_count)
                with col2:
                    st.metric("Incorrect", incorrect_count)
                with col3:
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Provide feedback
                if accuracy >= 90:
                    st.success("🎉 Excellent accuracy! The extractor is performing very well.")
                elif accuracy >= 75:
                    st.info("👍 Good accuracy. Consider fine-tuning parameters for better results.")
                elif accuracy >= 60:
                    st.warning("⚠️ Moderate accuracy. The extractor may need significant improvements.")
                else:
                    st.error("❌ Low accuracy. Consider reviewing the extraction logic.")
    
    def run(self):
        """Run the Streamlit app"""
        self.render_header()
        
        # Sidebar settings
        settings = self.render_sidebar()
        
        # File upload
        st.subheader("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to extract its structure"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{len(uploaded_file.getvalue())} bytes"
            }
            st.json(file_details)
            
            # Process file
            if st.button("🔍 Extract Structure", type="primary"):
                result = self.process_uploaded_file(uploaded_file, settings)
                
                if result:
                    # Store results in session state for persistence
                    st.session_state['extraction_result'] = result
                    st.session_state['extraction_settings'] = settings
                    st.rerun()
        
        # Display results if available
        if 'extraction_result' in st.session_state:
            st.divider()
            self.render_extraction_results(
                st.session_state['extraction_result'],
                st.session_state.get('extraction_settings', settings)
            )
            
            # Download results
            if st.button("💾 Download Results as JSON"):
                result_json = json.dumps(st.session_state['extraction_result'], indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=result_json,
                    file_name=f"pdf_structure_{int(time.time())}.json",
                    mime="application/json"
                )

# Import required modules at the top
import re

def main():
    """Main function to run the app"""
    validator = StreamlitPDFValidator()
    validator.run()

if __name__ == "__main__":
    main()