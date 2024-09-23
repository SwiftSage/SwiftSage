import gradio as gr
import openai
import time
import re
import os

# Available models
MODELS = [
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct"
]

# Sambanova API base URL
API_BASE = "https://api.sambanova.ai/v1"

def create_client():
    """Creates an OpenAI client instance."""
    openai.api_key = os.getenv("API_KEY")

    return openai.OpenAI(api_key=openai.api_key, base_url=API_BASE)

def chat_with_ai(message, chat_history, system_prompt):
    """Formats the chat history for the API call."""
    messages = [{"role": "system", "content": system_prompt}]
    for tup in chat_history:
        first_key = list(tup.keys())[0]  # First key
        last_key = list(tup.keys())[-1]   # Last key
        messages.append({"role": "user", "content": tup[first_key]})
        messages.append({"role": "assistant", "content": tup[last_key]})
    messages.append({"role": "user", "content": message})
    return messages

def respond(message, chat_history, model, system_prompt, thinking_budget):
    """Sends the message to the API and gets the response."""
    client = create_client()
    messages = chat_with_ai(message, chat_history, system_prompt.format(budget=thinking_budget))
    start_time = time.time()

    try:
        completion = client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        thinking_time = time.time() - start_time
        return response, thinking_time
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, time.time() - start_time

def parse_response(response):
    """Parses the response from the API."""
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    reflection_match = re.search(r'<reflection>(.*?)</reflection>', response, re.DOTALL)

    answer = answer_match.group(1).strip() if answer_match else ""
    reflection = reflection_match.group(1).strip() if reflection_match else ""
    steps = re.findall(r'<step>(.*?)</step>', response, re.DOTALL)

    if answer == "":
        return response, "", ""

    return answer, reflection, steps

def generate(message, history, model, system_prompt, thinking_budget):
    """Generates the chatbot response."""
    response, thinking_time = respond(message, history, model, system_prompt, thinking_budget)

    if response.startswith("Error:"):
        return history + [({"role": "system", "content": response},)], ""

    answer, reflection, steps = parse_response(response)

    messages = []
    messages.append({"role": "user", "content": message})

    formatted_steps = [f"Step {i}: {step}" for i, step in enumerate(steps, 1)]
    all_steps = "\n".join(formatted_steps) + f"\n\nReflection: {reflection}"

    messages.append({"role": "assistant", "content": all_steps, "metadata": {"title": f"Thinking Time: {thinking_time:.2f} sec"}})
    messages.append({"role": "assistant", "content": answer})

    return history + messages, ""

# Define the default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant in normal conversation.
When given a problem to solve, you are an expert problem-solving assistant. 
Your task is to provide a detailed, step-by-step solution to a given question. 
Follow these instructions carefully:
1. Read the given question carefully and reset counter between <count> and </count> to {budget}
2. Generate a detailed, logical step-by-step solution.
3. Enclose each step of your solution within <step> and </step> tags.
4. You are allowed to use at most {budget} steps (starting budget), 
   keep track of it by counting down within tags <count> </count>, 
   STOP GENERATING MORE STEPS when hitting 0, you don't have to use all of them.
5. Do a self-reflection when you are unsure about how to proceed, 
   based on the self-reflection and reward, decides whether you need to return 
   to the previous steps.
6. After completing the solution steps, reorganize and synthesize the steps 
   into the final answer within <answer> and </answer> tags.
7. Provide a critical, honest and subjective self-evaluation of your reasoning 
   process within <reflection> and </reflection> tags.
8. Assign a quality score to your solution as a float between 0.0 (lowest 
   quality) and 1.0 (highest quality), enclosed in <reward> and </reward> tags.
Example format:            
<count> [starting budget] </count>
<step> [Content of step 1] </step>
<count> [remaining budget] </count>
<step> [Content of step 2] </step>
<reflection> [Evaluation of the steps so far] </reflection>
<reward> [Float between 0.0 and 1.0] </reward>
<count> [remaining budget] </count>
<step> [Content of step 3 or Content of some previous step] </step>
<count> [remaining budget] </count>
...
<step>  [Content of final step] </step>
<count> [remaining budget] </count>
<answer> [Final Answer] </answer> (must give final answer in this format)
<reflection> [Evaluation of the solution] </reflection>
<reward> [Float between 0.0 and 1.0] </reward>
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Llama3.1-Instruct-o1")
    # gr.Markdown("[Powered by SambaNova Cloud, Get Your API Key Here](https://cloud.sambanova.ai/apis)")

    # with gr.Row():
    #     api_key = gr.Textbox(label="API Key", type="password", placeholder="(Optional) Enter your API key here for more availability")

    with gr.Row():
        model = gr.Dropdown(choices=MODELS, label="Select Model", value=MODELS[0])
        thinking_budget = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Thinking Budget", info="maximum times a model can think")

    chatbot = gr.Chatbot(label="Chat", show_label=False, show_share_button=False, show_copy_button=True, likeable=True, layout="panel", type="messages")

    msg = gr.Textbox(label="Type your message here...", placeholder="Enter your message...")

    gr.Button("Clear Chat").click(lambda: ([], ""), inputs=None, outputs=[chatbot, msg])

    system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=15, interactive=True)

    msg.submit(generate, inputs=[msg, chatbot, model, system_prompt, thinking_budget], outputs=[chatbot, msg])

demo.launch(share=True, show_api=False) 