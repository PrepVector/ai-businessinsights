import json
import re
def extract_json_from_response(response_text):
    """
    Extract JSON from a LLM text response that might contain additional text.
    """
    try:
        # Attempt to find JSON enclosed in triple backticks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Attempt to find JSON enclosed in curly braces
        json_match = re.search(r'(\{[\s\S]*\})', response_text)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Attempt to parse the entire response
        return json.loads(response_text)
    
    except json.JSONDecodeError as e:
        print(f"JSON extraction failed: {e}")
        return None

def format_questions_to_text(questions_dict: dict) -> str:
    """Converts the nested questions dictionary to a formatted text for editing."""
    text = ""
    for key, value in questions_dict.items():
        category_name = value.get("category", key).strip()
        text += f"### {category_name}\n"
        for idx, question in enumerate(value.get("questions", []), 1):
            text += f"{idx}. {question}\n"
        text += "\n"
    return text

def parse_text_to_questions(text: str) -> dict:
    """Parses edited text back into the nested dictionary format."""
    lines = text.split('\n')
    result = {}
    current_key = None
    current_category = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('###'):
            current_category = line[3:].strip()
            current_key = current_category.lower().replace(' ', '_')
            result[current_key] = {
                "category": current_category,
                "questions": []
            }
        elif line[0].isdigit() and '. ' in line and current_key:
            question = line.split('. ', 1)[1]
            result[current_key]["questions"].append(question)

    return result