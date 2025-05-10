# InsightBot
An intelligent data analysis system combining Business Analytics and Exploratory Data Analysis capabilities. (# ai-agent-insight-generation ->
AI agent that generates data insights)

![InsightBot_Workflow drawio](https://github.com/user-attachments/assets/4acac482-eb8c-4f10-8362-5e66737f4769)

## Overview
InsightBot is a sophisticated analysis tool built on the CrewAI framework that leverages two specialized agents:
- **BA Agent**: Handles business analytics and insights generation
- **EDA Agent**: Performs exploratory data analysis and statistical computations

InsightBot harnesses the power of CrewAI, a multi-agent framework designed for orchestrating role-playing, autonomous AI agents. This framework enables complex task delegation and collaboration between the BA and EDA agents, allowing them to work together seamlessly to provide comprehensive data analysis and business insights.

## Features
- Automated data analysis and visualization
- Business insights generation
- Statistical analysis and pattern detection
- Interactive query processing
- Comprehensive data exploration
- Multi-agent collaboration powered by CrewAI framework

## Getting Started
### Prerequisites
1. Create a virtual environment (3.11 recommended)
```bash
python3.11 -m venv venv
```
2. Activate the virtual environment
```bash
venv\Scripts\activate
```
3. Install the required dependencies
```bash
pip install -r requirements.txt
```
### Usage
1. Run the main application:
```bash
streamlit run app.py
```
2. Use the system through the provided interface to:
    - Load and analyze datasets (The uploaded dataset, metdata (optional) is stored in the datapath_info folder. It can be deleted if required (Delete Dataset button)
    - Generate business insights (Generate EDA Questions button)
    - Perform exploratory analysis and Visualize data patterns (Run EDA Analysis button)

## Components
### CrewAI Framework
InsightBot is built on CrewAI, an advanced multi-agent framework that:
- Enables role-based agent task allocation
- Provides sophisticated inter-agent communication protocols
- Manages agent workflows and task dependencies
- Facilitates autonomous decision-making and problem-solving
### BA Agent
- Business analytics processing
- Insight generation
- Strategic recommendations
- CrewAI-driven task delegation and coordination
### EDA Agent
- Batch processing of questions for each category
- Rate limit handling mechnaism implemented
- Custom Python REPL Tool for executing python code (see tools folder)
- Statistical analysis
- Data visualization
- Pattern detection
- Exploratory analysis
- Collaborative analysis with BA Agent via CrewAI protocols

## Further Improvements
- Generate Business Report (Note: It will in bullet pointers -> Comprehensive report containing the summary of the EDA on dataset)
- Dockefile
- API Integration
- Deployment to Streamlit Cloud etc
- Enhanced agent collaboration features
- Extended CrewAI capabilities
- Experimentation with paid and other open-source API (apart from Groq)

## Limitations
- Rate Limit Error
- Optimize the prompt templates for improving results (Sometimes reports are coming empty, or plots not getting generated (but code generation is present)
- Continue/ Run multiple times with less number of questions to see accurate results.
- CrewAI framework dependencies and compatibility considerations

## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under the MIT License.
