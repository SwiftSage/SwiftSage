import json
import re
import hjson

def extract_from_json(json_string, keys_to_extract):
    # Try to parse JSON normally first
    try:
        data = hjson.loads(json_string)
        return {key: data.get(key) for key in keys_to_extract}
    except json.JSONDecodeError:
        # If JSON is invalid, use regex for fuzzy matching
        result = {}
        for key in keys_to_extract:
            pattern = r'"{}"\\s*:\\s*(\\[.*?\\]|\\{{.*?\\}}|".*?")'.format(re.escape(key))
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                value = match.group(1)
                # Try to parse the value
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # If parsing fails, use the raw string
                    parsed_value = value.strip('"')
                result[key] = parsed_value
        return result

# Example usage:
json_string = '''
{
  "reasoning_steps": [
    "Step 1: Express g(x) in terms of h(x).",
    "Let g(h(x)) = x^2 - 8x + 10. We can substitute h(x) = x - 4 into this expression.",
    "g(x - 4) = x^2 - 8x + 10",
    "Step 2: Manipulate the expression for g(h(x)) to isolate g(x).",
    "To isolate g(x), we need to express g(x) in terms of x, not h(x).",
    "Let y = x - 4. Then, x = y + 4.",
    Substitute this into the expression for g(h(x)).",
    "g(y) = (y + 4)^2 - 8(y + 4) + 10",
    "Expand the expression for g(y).",
    "g(y) = y^2 + 8y + 16 - 8y - 32 + 10",
    "Simplify the expression for g(y).",
    "g(y) = y^2 - 6",
    "Step 3: Replace y with x to get the final expression for g(x).",
    "g(x) = x^2 - 6"
  ],
  "final_answer": "x^2 - 6"
}
'''
 
keys_to_extract = ["reasoning_steps", "final_answer"]
result = extract_from_json(json_string, keys_to_extract)
print(result)
