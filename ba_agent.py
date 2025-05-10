from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import pandas as pd
import os
import warnings
import json
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from utils import extract_json_from_response
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Initialize the language model
llm = ChatGroq(
    temperature=0,
    model_name="groq/llama3-70b-8192"
)

def generate_summary_text(df):
    """Generate comprehensive summary statistics for the dataset"""
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nGenerating dataset summary statistics...")
    
    # Basic summary statistics
    summary_stats = df.describe().to_dict()
    # Count missing values
    missing_values = df.isnull().sum().to_dict()
    # Count duplicate rows
    duplicate_count = df.duplicated().sum()
    # Get data types
    data_types = df.dtypes.apply(lambda x: str(x)).to_dict()
    
    # Advanced statistical analysis
    numerical_stats = {}
    categorical_stats = {}

    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            # Numerical statistics
            numerical_stats[column] = {
                'mean': df[column].mean(),
                'median': df[column].median(),
                'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'std': df[column].std(),
                'variance': df[column].var(),
                'iqr': df[column].quantile(0.75) - df[column].quantile(0.25),
                'skewness': df[column].skew(),
                'kurtosis': df[column].kurtosis()
            }
        else:
            # Categorical statistics
            categorical_stats[column] = {
                'unique_values': df[column].nunique()
            }

    # Generate summary text from the statistics
    summary_text_llm = f"""
    Dataset Summary:
    - {df.shape[0]} rows and {df.shape[1]} columns
    - Column names: {', '.join(df.columns.tolist())}
    - Missing values: {sum(missing_values.values())} in total
    - Duplicate rows: {duplicate_count}
    - Data types: {len([dt for dt in data_types.values() if 'float' in dt])} numerical columns, {len([dt for dt in data_types.values() if 'object' in dt])} categorical columns
    - Summary statistics: {summary_stats}
    - Advanced numerical statistics: {numerical_stats}
    - Categorical statistics: {categorical_stats}
    """
    return summary_text_llm

def run_business_analysis(df: pd.DataFrame, metadata: str = None) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate EDA questions across five categories, based on metadata and dataset summary.
    Uses CrewAI to manage the business analysis process.
    
    Args:
        df: The pandas DataFrame to analyze
        metadata: Optional string containing dataset metadata/context
        
    Returns:
        A nested dict with EDA questions organized by category
    """
    # Get dataset summary
    summary = generate_summary_text(df)
    has_metadata = metadata is not None and metadata.strip() != ""
    
    # Define the Business Analyst agent with more specific instructions
    business_analyst = Agent(
        role="Business Analyst",
        goal="Generate comprehensive exploratory data analysis questions in JSON format",
        backstory="An expert business analyst specialized in formulating insightful questions for data exploration and returning them in JSON format.",
        llm=llm,
        verbose=True
    )
    
    # Create task description based on whether metadata is provided
    if has_metadata:
        task_description = f"""
        You are tasked with generating exploratory data analysis (EDA) questions for a dataset.
        
        Metadata:
        {metadata}
        
        Dataset Summary:
        {summary}
        
        Generate exactly 5 relevant questions for EACH of these categories:
        1. Data Quality Assessment - Questions about data completeness, validity, consistency
        2. Statistical Summary - Questions about distributions, central tendency, spread
        3. Outlier Detection - Questions about detecting anomalies or extreme values
        4. Feature Relationships - Questions about correlations or interactions between features
        5. Pattern Trend Anomalies - Questions about trends, seasonality, and unexpected shifts
        
        CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON with exactly these keys:
        - data_quality_assessment
        - statistical_summary
        - outlier_detection
        - feature_relationships
        - pattern_trend_anomalies
        
        Each key should contain a list of 5 relevant questions as strings.
        DO NOT include any explanatory text, just the JSON.
        DO NOT say "I can give a great answer" or similar phrases.
        ONLY respond with the JSON object.
        """
    else:
        task_description = f"""
        You are tasked with generating exploratory data analysis (EDA) questions for a dataset.
        
        Dataset Summary:
        {summary}
        
        Generate exactly 5 relevant questions for EACH of these categories:
        1. Data Quality Assessment - Questions about data completeness, validity, consistency
        2. Statistical Summary - Questions about distributions, central tendency, spread
        3. Outlier Detection - Questions about detecting anomalies or extreme values
        4. Feature Relationships - Questions about correlations or interactions between features
        5. Pattern Trend Anomalies - Questions about trends, seasonality, and unexpected shifts
        
        CRITICAL INSTRUCTION: You MUST respond with ONLY a valid JSON with exactly these keys:
        - data_quality_assessment
        - statistical_summary
        - outlier_detection
        - feature_relationships
        - pattern_trend_anomalies
        
        Each key should contain a list of 5 relevant questions as strings.
        DO NOT include any explanatory text, just the JSON.
        DO NOT say "I can give a great answer" or similar phrases.
        ONLY respond with the JSON object.
        """
    
    # Create the task with more specific instructions
    generate_eda_questions_task = Task(
        description=task_description,
        agent=business_analyst,
        expected_output="ONLY a valid JSON containing 5 questions for each of the 5 categories (25 questions total)"
    )
    
    # First attempt with CrewAI
    crew = Crew(
        agents=[business_analyst],
        tasks=[generate_eda_questions_task],
        process=Process.sequential,
        verbose=True
    )
    
    crew_output = crew.kickoff()
    print(f"Business analyst output: {crew_output}")
    
    # Extract JSON from the response
    json_data = extract_json_from_response(str(crew_output))
    
    if json_data:
        # Validate required keys
        required_keys = ["data_quality_assessment", "statistical_summary", 
                         "outlier_detection", "feature_relationships", 
                         "pattern_trend_anomalies"]
        
        if all(key in json_data for key in required_keys):
            # Convert to the required nested format
            nested_dict = {
                key: {
                    "category": key.replace("_", " ").title(),
                    "questions": questions
                }
                for key, questions in json_data.items()
                if key in required_keys
            }
            
            # Save the nested dictionary to a JSON file
            output_dir = "ba_agent_output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"eda_questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(output_path, "w") as outfile:
                json.dump(nested_dict, outfile, indent=4)
                
            print(f"Business analysis questions generated at: {output_path}")
            return nested_dict
        else:
            missing_keys = [key for key in required_keys if key not in json_data]
            print(f"Error: JSON response missing required keys: {missing_keys}")
            return None
    else:
        print("Error: Could not extract valid JSON from the response")
        return None