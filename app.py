import gradio as gr
import os
import json
import logging
import numpy as np
from utils import (PromptTemplate, api_configs, setup_logging)
from data_loader import load_data
from evaluate import evaluate
from main import SwiftSage, run_test, run_benchmark
import multiprocessing 



def solve_problem(problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, reward_model_id, use_retrieval, start_with_sage):
    # Configuration for each LLM
    swift_config = {
        "model_id": swift_model_id,
        "api_config": api_configs['Together']
    }

    reward_config = {
        "model_id": reward_model_id,
        "api_config": api_configs['Together']
    }

    sage_config = {
        "model_id": sage_model_id,
        "api_config": api_configs['Together']
    }

    # specify the path to the prompt templates
    prompt_template_dir = './prompt_templates'
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

    reasoning, solution = s2.solve(problem, max_iterations, reward_threshold)
    return reasoning, solution

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SwiftSage Testing Interface")

    with gr.Row():
        swift_model_id = gr.Textbox(label="Swift Model ID", value="meta-llama/Meta-Llama-3-8B-Instruct-Turbo")
        reward_model_id = gr.Textbox(label="Reward Model ID", value="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
        sage_model_id = gr.Textbox(label="Sage Model ID", value="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo")

    with gr.Row():
        max_iterations = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Max Iterations")
        reward_threshold = gr.Slider(minimum=4, maximum=10, value=8, step=1, label="Reward Threshold")

    use_retrieval = gr.Checkbox(label="Use Retrieval Augmentation", value=False)
    start_with_sage = gr.Checkbox(label="Start with Sage", value=False)

    problem = gr.Textbox(label="Problem", placeholder="Enter the problem to solve...")

    solve_button = gr.Button("Solve Problem")
    reasoning_output = gr.Textbox(label="Reasoning", interactive=False)
    solution_output = gr.Textbox(label="Solution", interactive=False)

    solve_button.click(
        solve_problem,
        inputs=[problem, max_iterations, reward_threshold, swift_model_id, sage_model_id, reward_model_id, use_retrieval, start_with_sage],
        outputs=[reasoning_output, solution_output]
    )

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    demo.launch(share=False, show_api=False)
