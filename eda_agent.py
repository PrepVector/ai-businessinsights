from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
import warnings
import os
import re
import time
import random
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Import the custom PythonREPLTool
from tools.custom_tool import PythonREPLTool

# Define a custom exception for rate limits that will be used consistently
class RateLimitException(Exception):
    pass
class QuestionList(BaseModel):
    questions: List[str]

# Decorator to retry on RateLimitError with exponential backoff
@retry(
    retry=retry_if_exception_type((RateLimitException, Exception)),
    wait=wait_exponential(multiplier=2, min=10, max=120),  # Increased backoff times
    stop=stop_after_attempt(8)  # Increased max attempts
)
def create_llm_with_retry():
    """Create an LLM instance with retry logic for rate limits"""
    try:
        # Try primary model
        llm = ChatGroq(
            temperature=0.01,
            model_name="groq/llama-3.3-70b-versatile",
            # model_name="groq/llama3-70b-8192",
            # max_tokens=600,

            # model_name = "groq/mixtral-8x7b-32768",
            # max_tokens=8000
        )
        # Return the LLM instance
        return llm
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a rate limit error
        if "rate limit" in error_str or "too many requests" in error_str:
            print(f"Rate limit hit: {str(e)}")
            # Add jitter to avoid synchronized retries
            wait_time = 20 + random.uniform(2, 10)
            print(f"Waiting for {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)
            raise RateLimitException(f"Rate limit exceeded: {str(e)}")
        else:
            # For other errors
            print(f"Error creating LLM: {str(e)}")
            wait_time = 10 + random.uniform(1, 5)
            print(f"Waiting for {wait_time:.2f} seconds before retry...")
            time.sleep(wait_time)
            raise  # Re-raise to let tenacity handle the retry

def create_data_scientist():
    """Create the data scientist agent with improved retry handling"""
    try:
        # Create LLM with retry logic
        llm = create_llm_with_retry()
        
        # Create the data scientist agent
        return Agent(
            role="Data Scientist",
            goal="Generate clean, error-free code to answer multiple questions about data quality, execute the code, and provide clear interpretations with proper visualizations.",
            backstory="You are a skilled data scientist proficient in exploratory data analysis. You write clean, error-free code with proper string handling, formatting, and visualization management.",
            verbose=True,
            llm=llm,
            max_retry_limit=3  # Add retry limit for better error handling
        )
    except Exception as e:
        print(f"Error creating data scientist agent: {str(e)}")
        # If we fail to create the agent, try one more time after waiting
        wait_time = 30 + random.uniform(5, 15)
        print(f"Waiting {wait_time:.2f} seconds before retry...")
        time.sleep(wait_time)
        
        # Create LLM with retry logic again
        llm = create_llm_with_retry()
        
        # Create the data scientist agent
        return Agent(
            role="Data Scientist",
            goal="Generate clean, error-free code to answer multiple questions about data quality, execute the code, and provide clear interpretations with proper visualizations.",
            backstory="You are a skilled data scientist proficient in exploratory data analysis. You write clean, error-free code with proper string handling, formatting, and visualization management.",
            verbose=True,
            llm=llm,
            max_retry_limit=3
        )

def create_batch_eda_task(questions_list, category_name, datapath_info, imagepath_dir):
    """Create a task for the data scientist to analyze a batch of questions in one category"""
    questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
    
    return Task(
        description=f"""
        You are tasked with generating the Python code to answer multiple questions for the category: {category_name}. 
        
        Dataset information:
        - Dataset path: {datapath_info}
        - Image save path: {imagepath_dir}
        
        Questions to answer:
        {questions_str}
        
        IMPORTANT: Follow these guidelines EXACTLY:
        
        1. Generate ONE Python script that analyzes ALL questions for this category
        2. Your code MUST:
           - Include proper imports for pandas, numpy, matplotlib, seaborn as needed
           - Load the dataset from {datapath_info}
           - Create separate sections for each question with clear headers
           - Use appropriate analysis methods for each question
           - Print results to console
           - Save plots to {imagepath_dir} with descriptive filenames in the format: {imagepath_dir}/[category]_q[num]_[description].png
           - After saving each plot, explicitly print: "Plot saved to: [full_path_to_image]"
        
        3. For each plot:
           - Use plt.savefig('{imagepath_dir}/[category]_q[num]_[description].png', bbox_inches='tight', dpi=300)
           - ALWAYS print: "Plot saved to: [full_path_to_image]" immediately after saving
           - Call plt.close() to free memory
        
        4. Ensure your code:
           - Has NO syntax errors
           - Uses proper string formatting (no unterminated strings)
           - Handles errors gracefully
           - Is well-commented
           - Properly saves all plots and prints their save paths
        
        5. Structure your code like this:
           ```python
           # Import necessary libraries
           import pandas as pd
           import numpy as np
           import matplotlib.pyplot as plt
           import seaborn as sns
           
           # Load dataset
           df = pd.read_csv('{datapath_info}')
           
           # Question 1
           print("==== Question 1 Analysis ====")
           # Your analysis code here
           
           # Question 2
           print("==== Question 2 Analysis ====")
           # Your analysis code here
           
           # And so on...
           ```
        """,
        expected_output="""
    Provide a structured analysis with:
    
    # {category_name} Analysis
    
    ### Question 1
    - [Question text]
    
    #### Code
    ```python
    [Clean, properly formatted Python code]
    ```
    
    #### Code Output
    ```
    [Results from execution]
    ```
    
    #### Detailed Analysis
    [Clear detailed interpretation of results]
    
    #### Plots Generated
    [List of plots with full paths as printed in the code output]
    
    ### Question 2
    [Same structure as above]
    
    ... and so on for all questions
    """,
        agent=create_data_scientist()
    )
    
def extract_plot_paths(result_text):
    """Extract plot file paths from the result text"""
    plot_paths = []
    patterns = [
        r'Plot saved to: ([\w/\\\.]+\.png)',
        r'Saved to: ([\w/\\\.]+\.png)',
        r'(eda_agent_report/images/[\w\-_\.]+\.png)',
        r'saving plot to: ([\w/\\\.]+\.png)',
        r'Saved plot to: ([\w/\\\.]+\.png)',
        r'(eda_agent_report\\images\\[\w\-_\.]+\.png)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, result_text, re.IGNORECASE)
        plot_paths.extend(matches)
    
    # Clean up paths (remove duplicates, fix slashes)
    cleaned_paths = []
    for path in plot_paths:
        path = path.replace('\\', '/')
        if path not in cleaned_paths:
            cleaned_paths.append(path)
    
    return cleaned_paths

def run_eda_analysis(dataset_path, questions_data, imagepath_dir):
    """Run the EDA crew with the given questions - batch by category with improved rate limit handling"""
    
    # Create directories if they don't exist
    os.makedirs(imagepath_dir, exist_ok=True)
    os.makedirs("eda_agent_report", exist_ok=True)
    
    # Create PythonREPLTool
    python_repl_tool = PythonREPLTool()
    
    all_results = {}
    all_category_reports = {}
    
    # First attempt to establish a connection to verify API works
    print("Verifying API connection before starting analysis...")
    try:
        # Initial connection test with forced delay to warm up
        time.sleep(5)  # Initial cooldown
        llm = create_llm_with_retry()
        print("API connection established successfully!")
        time.sleep(10)  # Additional cooldown after successful connection
    except Exception as e:
        print(f"Warning: Initial API connection test failed: {str(e)}")
        print("Will attempt to proceed with analysis anyway...")
        time.sleep(30)  # Extended cooldown after failure
    
    # Process each category
    for category_idx, (category, category_data) in enumerate(questions_data.items()):
        print(f"\n\n===== Processing category {category_idx+1}/{len(questions_data)}: {category_data['category']} =====")
        category_report = f"# {category_data['category']} Report\n\n"
        
        # Add a delay between categories to prevent rate limits
        if category_idx > 0:
            wait_time = 90  # Increased wait time between categories
            print(f"Waiting {wait_time} seconds before processing next category...")
            time.sleep(wait_time)
        
        # Multiple attempts for processing a category
        max_category_attempts = 3
        for attempt in range(1, max_category_attempts + 1):
            try:
                # Create a fresh LLM instance for each category
                llm = create_llm_with_retry()
                
                # Create a batch task for all questions in this category
                task = create_batch_eda_task(
                    category_data["questions"], 
                    category_data['category'], 
                    dataset_path, 
                    imagepath_dir
                )
                
                # Create data scientist agent with PythonREPLTool and the fresh LLM
                data_scientist = create_data_scientist()
                data_scientist.tools = [python_repl_tool]
                task.agent = data_scientist
                
                # Create a crew with just this task
                crew = Crew(
                    agents=[data_scientist],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True
                )
                
                # Execute the crew with error handling and exponential backoff
                max_crew_attempts = 3
                for crew_attempt in range(1, max_crew_attempts + 1):
                    try:
                        print(f"Starting crew execution (attempt {crew_attempt}/{max_crew_attempts})...")
                        result = crew.kickoff()
                        
                        # Extract plot paths from the result
                        result_text = str(result)
                        plot_paths = extract_plot_paths(result_text)
                        
                        # Process and embed plot images in the report
                        if plot_paths:
                            # Split the result text into sections
                            sections = result_text.split('###')
                            updated_result = ""
                            
                            # Process each section
                            for section in sections:
                                if section.strip():
                                    updated_result += f"###{section}"
                                    
                                    # If this section contains "Plots Generated", add visualizations
                                    if "Plots Generated" in section and plot_paths:
                                        updated_result += "\n### Visualizations\n\n"
                                        for plot_path in plot_paths:
                                            # Clean up the path
                                            relative_path = plot_path.replace('\\', '/')
                                            if not relative_path.startswith('eda_agent_report'):
                                                relative_path = os.path.join('eda_agent_report', relative_path)
                                            
                                            updated_result += f"![Plot]({relative_path})\n\n"
                            
                            result_text = updated_result
                        
                        all_results[category] = {
                            "questions": category_data["questions"],
                            "result": result_text,
                            "plots": plot_paths
                        }
                        
                        category_report += f"{result_text}\n\n"
                        
                        print(f"Completed batch analysis for category: {category_data['category']}")
                        
                        # Successfully completed this category, break both loops
                        break
                        
                    except Exception as e:
                        error_str = str(e).lower()
                        if crew_attempt < max_crew_attempts:
                            # Check if it's a rate limit error
                            if "rate limit" in error_str or "too many requests" in error_str:
                                wait_time = 120 * crew_attempt  # Progressive backoff
                                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                            else:
                                wait_time = 60 * crew_attempt
                                print(f"Error during crew execution: {str(e)}")
                                print(f"Waiting {wait_time} seconds before retry...")
                            
                            # Add jitter to avoid synchronized retries
                            wait_time += random.uniform(5, 15)
                            time.sleep(wait_time)
                        else:
                            # Last attempt failed, re-raise the exception
                            raise
                
                # If we made it here, we've successfully processed this category
                break
                    
            except Exception as e:
                error_msg = f"Error processing category: {category_data['category']}. Error: {str(e)}"
                print(f"ERROR: {error_msg}")
                
                if attempt < max_category_attempts:
                    retry_wait = 60 * attempt  # Increase wait time with each attempt
                    # Add jitter to avoid synchronized retries
                    retry_wait += random.uniform(5, 15)
                    print(f"Waiting {retry_wait:.2f} seconds before retrying category...")
                    time.sleep(retry_wait)
                else:
                    # All attempts failed, add error message to report
                    category_report += f"### Error\nFailed after {max_category_attempts} attempts: {error_msg}\n\n"
                    all_results[category] = {
                        "questions": category_data["questions"],
                        "result": f"Error: {str(e)}",
                        "plots": []
                    }
        
        all_category_reports[category] = category_report
        
        # Save individual category report
        report_filename = f"eda_agent_report/{category.lower().replace(' ', '_')}_report.md"
        with open(report_filename, "w", encoding='utf-8') as f:
            f.write(category_report)
        
        print(f"Category report saved to {report_filename}")
    
    # Generate combined final technical report
    final_report = "# Exploratory Data Analysis Technical Report\n\n"
    final_report += "## Executive Summary\n\n"
    final_report += "This report presents a comprehensive exploratory data analysis with generated visualizations.\n\n"
    final_report += "## Table of Contents\n\n"
    
    for category, category_data in questions_data.items():
        if category in all_category_reports:
            final_report += f"- [{category_data['category']}](#{category_data['category'].lower().replace(' ', '-')})\n"
    
    final_report += "\n"
    
    # Add each category report
    for category, category_data in questions_data.items():
        if category in all_category_reports:
            final_report += f"## {category_data['category']}\n\n"
            final_report += all_category_reports[category]
            final_report += "---\n\n"
    
    # Save final report
    final_report_path = "eda_agent_report/technical_report.md"
    with open(final_report_path, "w", encoding='utf-8') as f:
        f.write(final_report)
    
    print(f"Final technical report saved to {final_report_path}")    
    return final_report