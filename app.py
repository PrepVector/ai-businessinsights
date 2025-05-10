import streamlit as st
import pandas as pd
import os
import shutil
import re
from utils import format_questions_to_text, parse_text_to_questions 
from ba_agent import run_business_analysis
from eda_agent import run_eda_analysis
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)

# Create datapath_info directory if it doesn't exist
def ensure_data_directory():
    if not os.path.exists("datapath_info"):
        os.makedirs("datapath_info")

def save_uploaded_file(uploaded_file):
    ensure_data_directory()
    file_path = os.path.join("datapath_info", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def extract_detailed_analysis(technical_report):
    """Extract all 'Detailed Analysis' sections from the technical report using regex."""
    pattern = r"#### Detailed Analysis\n(.*?)(?=\n#### |$)"
    matches = re.findall(pattern, technical_report, re.DOTALL)
    return matches

def generate_business_report(detailed_analysis_sections):
    """Generate a business report with key insights from detailed analysis sections."""
    # Combine all detailed analysis sections
    combined_analysis = "\n".join(detailed_analysis_sections)
    
    prompt = f"""Based on the following detailed analysis from a technical report, 
    create clear business insights in bullet point format. Focus on actionable 
    insights and key findings:

    {combined_analysis}
    
    Please format the output as bullet points starting with '- '
    NOTE: Only use information directly from the analysis; do not introduce new data 
or make assumptions.
    """
    
    try:
        response = llm.invoke(prompt,temperature=0)
        result = response.content
        # Extract just the bullet points from the response
        bullet_points = re.findall(r'- .*', result)
        
        # Format the business report
        business_report = "# Business Insights Report\n\n"
        business_report += "## Key Insights\n\n"
        for point in bullet_points:
            business_report += f"{point}\n"        
        return business_report
    
    except Exception as e:
        return f"Error generating business report: {str(e)}"

def main():
    st.set_page_config(page_title="InsightBot", page_icon="🤖", layout="wide")
    st.title("InsightBot")
    st.markdown("<h3 style='color: green;'>AI-powered Exploratory Data Analysis Assistant</h3>", unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'questions_dict' not in st.session_state:
        st.session_state.questions_dict = None
    if 'questions_text' not in st.session_state:
        st.session_state.questions_text = ""
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = False
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = None
    if 'metadata_path' not in st.session_state:
        st.session_state.metadata_path = None
    if 'technical_report' not in st.session_state:
        st.session_state.technical_report = None
    
    # Sidebar elements
    st.sidebar.title("Data Controls")
    
    # File uploaders for dataset and metadata
    dataset_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=['csv'])
    metadata_file = st.sidebar.file_uploader("Upload your metadata (TXT file) (Optional)", type=['txt'])
    
    # Add a delete dataset button to the sidebar
    if st.sidebar.button("Delete Dataset"):
        if os.path.exists("datapath_info"):
            shutil.rmtree("datapath_info")
            st.session_state.dataset_path = None
            st.session_state.metadata_path = None
            st.success("Dataset deleted successfully!")
            st.rerun()
    
    # Add a reload button to the sidebar
    if st.sidebar.button("Reload Page"):
        st.session_state.clear()
        st.rerun()
    
    st.sidebar.markdown("### Reload the page to reset the state.")
    
    # Handle file uploads
    if dataset_file and st.session_state.dataset_path is None:
        st.session_state.dataset_path = save_uploaded_file(dataset_file)
    
    if metadata_file and st.session_state.metadata_path is None:
        st.session_state.metadata_path = save_uploaded_file(metadata_file)
    
    # Load and display data if it exists
    if st.session_state.dataset_path and os.path.exists(st.session_state.dataset_path):
        # Load dataset
        dataset = pd.read_csv(st.session_state.dataset_path)
        
        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.write("\n")
            st.dataframe(dataset.head())
        
        # Load metadata
        metadata = ""
        if st.session_state.metadata_path and os.path.exists(st.session_state.metadata_path):
            with open(st.session_state.metadata_path, 'r') as f:
                metadata = f.read()
        
        with col2:
            st.subheader("Metadata")
            st.text_area("Metadata content", metadata, height=250, disabled=True, label_visibility="collapsed")
        
        # Generate EDA questions
        if st.button("Generate EDA Questions") or st.session_state.generated_questions:
            if not st.session_state.generated_questions:
                with st.spinner("Generating questions..."):
                    st.session_state.questions_dict = run_business_analysis(dataset, metadata)
                    st.session_state.questions_text = format_questions_to_text(st.session_state.questions_dict)
                    st.session_state.generated_questions = True
                st.success("EDA questions generated successfully!")
            
            # Editable text area for questions
            edited_questions = st.text_area("Edit your EDA questions below", value=st.session_state.questions_text, height=400)
            
            # Update the session state if the questions were edited
            if edited_questions != st.session_state.questions_text:
                st.session_state.questions_text = edited_questions

            imagepath_dir = "eda_agent_report/images"
            
            # Button to run EDA analysis with the edited questions
            if st.button("Run EDA Analysis"):
                with st.spinner("Running EDA analysis..."):
                    # Parse the edited text back to structured format
                    updated_questions = parse_text_to_questions(st.session_state.questions_text)
                    # Run the EDA analysis
                    results = run_eda_analysis(st.session_state.dataset_path, updated_questions, imagepath_dir)
                    st.session_state.technical_report = results
                    if st.button("Stop Analysis"):
                        st.warning("Analysis stopped by user")
                        st.stop()
                    st.success("EDA analysis completed!")
                    st.download_button(
                        label="Download Full Technical Report as Markdown",
                        data=results,
                        file_name="eda_agent_report/technical_report.md",
                        mime="text/markdown"
                    )
            
            if st.session_state.technical_report:
                if st.button("Generate Business Report"):
                    with st.spinner("Extracting insights and generating business report..."):
                        detailed_analysis = extract_detailed_analysis(st.session_state.technical_report)                        
                        business_report = generate_business_report(detailed_analysis)
                        
                        st.success("Business report generated successfully!")
                        st.markdown(business_report)
                        
                        st.download_button(
                            label="Download Business Report as Markdown",
                            data=business_report,
                            file_name="eda_agent_report/business_report.md",
                            mime="text/markdown"
                        )
    else:
        st.info("Please upload a dataset file to begin.")

if __name__ == "__main__":
    main()
