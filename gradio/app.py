import json
import logging
import multiprocessing
import os

import gradio as gr

from swiftsage.agents import SwiftSage
from swiftsage.utils.commons import PromptTemplate, api_configs, setup_logging
from pkg_resources import resource_filename

def solve_problem(problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, feedback_model_id, use_retrieval, start_with_sage):
    # Configuration for each LLM
    max_iterations = int(max_iterations)
    reward_threshold = int(reward_threshold)

    swift_config = {
        "model_id": swift_model_id,
        "api_config": api_configs['Together']
    }

    reward_config = {
        "model_id": feedback_model_id,
        "api_config": api_configs['Together']
    }

    sage_config = {
        "model_id": sage_model_id,
        "api_config": api_configs['Together']
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
        reward_config,
        use_retrieval=use_retrieval,
        start_with_sage=start_with_sage,
    )

    reasoning, solution, messages = s2.solve(problem, max_iterations, reward_threshold)
    solution = solution.replace("Answer (from running the code):\n ", " ")
    return reasoning, solution


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # gr.Markdown("## SwiftSage: A Multi-Agent Framework for Reasoning")
    # use the html and center the title 
    gr.HTML("<h1 style='text-align: center;'>SwiftSage: A Multi-Agent Framework for Reasoning</h1>")

    with gr.Row(): 
        swift_model_id = gr.Textbox(label="😄 Swift Model ID", value="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo") 
        feedback_model_id = gr.Textbox(label="🤔 Feedback Model ID", value="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo") 
        sage_model_id = gr.Textbox(label="😎 Sage Model ID", value="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo") 
        # the following two should have a smaller width
    
    with gr.Accordion(label="⚙️ Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                max_iterations = gr.Textbox(label="Max Iterations", value="5")
                reward_threshold = gr.Textbox(label="Reward Threshold", value="8")
            # TODO: add top-p and temperature for each module for controlling 
            with gr.Column():
                top_p_swift = gr.Textbox(label="Top-p for Swift", value="0.9")
                temperature_swift = gr.Textbox(label="Temperature for Swift", value="0.7")
            with gr.Column():
                top_p_sage = gr.Textbox(label="Top-p for Sage", value="0.9")
                temperature_sage = gr.Textbox(label="Temperature for Sage", value="0.7")
            with gr.Column():
                top_p_reward = gr.Textbox(label="Top-p for Feedback", value="0.9")
                temperature_reward = gr.Textbox(label="Temperature for Feedback", value="0.7")

            use_retrieval = gr.Checkbox(label="Use Retrieval Augmentation", value=False, visible=False)
            start_with_sage = gr.Checkbox(label="Start with Sage", value=False, visible=False)

    problem = gr.Textbox(label="Input your problem", value="How many letter r are there in the sentence 'My strawberry is so ridiculously red.'?", lines=2)

    solve_button = gr.Button("🚀 Solve Problem")
    reasoning_output = gr.Textbox(label="Reasoning steps with Code", interactive=False)
    solution_output = gr.Textbox(label="Final answer", interactive=False)

    solve_button.click(
        solve_problem,
        inputs=[problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, feedback_model_id, use_retrieval, start_with_sage],
        outputs=[reasoning_output, solution_output]
    )


if __name__ == '__main__':
    # make logs dir if it does not exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    multiprocessing.set_start_method('spawn')
    demo.launch(share=False, show_api=False)
