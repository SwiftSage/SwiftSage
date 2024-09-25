import json
import logging
import multiprocessing
import os

import gradio as gr

from swiftsage.agents import SwiftSage
from swiftsage.utils.commons import PromptTemplate, api_configs, setup_logging
from pkg_resources import resource_filename

ENGINE = "Together"
# SWIFT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
SWIFT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct-Reference"
FEEDBACK_MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
SAGE_MODEL_ID = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

# ENGINE = "SambaNova"
# SWIFT_MODEL_ID = "Meta-Llama-3.1-8B-Instruct"
# FEEDBACK_MODEL_ID = "Meta-Llama-3.1-70B-Instruct"
# SAGE_MODEL_ID = "Meta-Llama-3.1-405B-Instruct"

def solve_problem(problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, feedback_model_id, use_retrieval, start_with_sage, swift_temperature, swift_top_p, sage_temperature, sage_top_p, feedback_temperature, feedback_top_p):
    global ENGINE
    # Configuration for each LLM
    max_iterations = int(max_iterations)
    reward_threshold = int(reward_threshold)

    swift_config = {
        "model_id": swift_model_id,
        "api_config": api_configs[ENGINE],
        "temperature": float(swift_temperature),
        "top_p": float(swift_top_p),
        "max_tokens": 2048,
    }

    feedback_config = {
        "model_id": feedback_model_id,
        "api_config": api_configs[ENGINE],
        "temperature": float(feedback_temperature),
        "top_p": float(feedback_top_p),
        "max_tokens": 2048,
    }

    sage_config = {
        "model_id": sage_model_id,
        "api_config": api_configs[ENGINE],
        "temperature": float(sage_temperature),
        "top_p": float(sage_top_p),
        "max_tokens": 2048,
    }

    # specify the path to the prompt templates
    # prompt_template_dir = './swiftsage/prompt_templates'
    # prompt_template_dir = resource_filename('swiftsage', 'prompt_templates')

    # Try multiple locations for the prompt templates
    possible_paths = [
        resource_filename('swiftsage', 'prompt_templates'),
        os.path.join(os.path.dirname(__file__), '..', 'swiftsage', 'prompt_templates'),
        os.path.join(os.path.dirname(__file__), 'swiftsage', 'prompt_templates'),
        '/app/swiftsage/prompt_templates',  # For Docker environments
    ]

    prompt_template_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            prompt_template_dir = path
            break

    dataset = [] 
    embeddings = [] # TODO: for retrieval augmentation (not implemented yet now)
    s2 = SwiftSage(
        dataset,
        embeddings,
        prompt_template_dir,
        swift_config,
        sage_config,
        feedback_config,
        use_retrieval=use_retrieval,
        start_with_sage=start_with_sage,
    )

    reasoning, solution, messages = s2.solve(problem, max_iterations, reward_threshold)
    solution = solution.replace("Answer (from running the code):\n ", " ")
    # generate HTML for the log messages and display them with wrap and a scroll bar and a max height in the code block with log style 

    log_messages = "<pre style='white-space: pre-wrap; max-height: 500px; overflow-y: scroll;'><code class='log'>" + "\n".join(messages) + "</code></pre>"
    return reasoning, solution, log_messages


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # gr.Markdown("## SwiftSage: A Multi-Agent Framework for Reasoning")
    # use the html and center the title 
    gr.HTML("<h1 style='text-align: center;'>SwiftSage: A General Reasoning Framework with Fast and Slow Thinking </h1> ")
    gr.HTML("<span>SwiftSage is a multi-agent reasoning framework that combines the strengths of different models for solving complex problems. It uses a Swift model for fast thinking, a Sage model for slow thinking, and a Feedback model for providing feedback and reward. More info is on our Github: <a style='color: gray' href='https://github.com/SwiftSage/SwiftSage'> https://github.com/SwiftSage/SwiftSage </a>. Contact: <a href='https://yuchenlin.xyz/'>Bill Yuchen Lin</a> </span>")

    with gr.Row(): 
        swift_model_id = gr.Textbox(label="😄 Swift Model ID", value=SWIFT_MODEL_ID)
        feedback_model_id = gr.Textbox(label="🤔 Feedback Model ID", value=FEEDBACK_MODEL_ID)
        sage_model_id = gr.Textbox(label="😎 Sage Model ID", value=SAGE_MODEL_ID)
        # the following two should have a smaller width
    
    with gr.Accordion(label="⚙️ Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                max_iterations = gr.Textbox(label="Max Iterations", value="5")
                reward_threshold = gr.Textbox(label="feedback Threshold", value="8")
            # TODO: add top-p and temperature for each module for controlling 
            with gr.Column():
                top_p_swift = gr.Textbox(label="Top-p for Swift", value="0.9")
                temperature_swift = gr.Textbox(label="Temperature for Swift", value="0.7")
            with gr.Column():
                top_p_sage = gr.Textbox(label="Top-p for Sage", value="0.9")
                temperature_sage = gr.Textbox(label="Temperature for Sage", value="0.7")
            with gr.Column():
                top_p_feedback = gr.Textbox(label="Top-p for Feedback", value="0.9")
                temperature_feedback = gr.Textbox(label="Temperature for Feedback", value="0.7")

            use_retrieval = gr.Checkbox(label="Use Retrieval Augmentation", value=False, visible=False)
            start_with_sage = gr.Checkbox(label="Start with Sage", value=False, visible=False)

    problem = gr.Textbox(label="Input your problem", value="How many letter r are there in the sentence 'My strawberry is so ridiculously red.'?", lines=2)

    solve_button = gr.Button("🚀 Solve Problem")
    reasoning_output = gr.Textbox(label="Reasoning steps with Code", interactive=False)
    solution_output = gr.Textbox(label="Final answer", interactive=False)

    # add a log display for showing the log messages
    with gr.Accordion(label="📜 Log Messages", open=False):
        log_output = gr.HTML("<p>No log messages yet.</p>")

    solve_button.click(
        solve_problem,
        inputs=[problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, feedback_model_id, use_retrieval, start_with_sage, temperature_swift, top_p_swift, temperature_sage, top_p_sage, temperature_feedback, top_p_feedback],
        outputs=[reasoning_output, solution_output, log_output],
    )



if __name__ == '__main__':
    # make logs dir if it does not exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    multiprocessing.set_start_method('spawn')
    demo.launch(share=False, show_api=False)
