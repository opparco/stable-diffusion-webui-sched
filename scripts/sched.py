from functools import partial, reduce

from modules import extra_networks, script_callbacks

from modules import prompt_parser
from modules.sd_hijack import model_hijack

import gradio as gr

def do_schedule(text, steps):

    clip = model_hijack.clip

    def get_prompt_lengths(text):
        batch_chunks, token_count = clip.process_texts([text])

        return token_count, clip.get_target_prompt_token_count(token_count)

    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)

    md = ''
    for step, prompt in flat_prompts:
        md += f'### step {step}\n'
        md += prompt
        md += f'\n'
        token_count, max_length = get_prompt_lengths(prompt)
        md += f'({token_count}/{max_length})\n'

    return md

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, variant="compact") as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", show_label=False, lines=3, placeholder="Prompt")
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling steps", value=20)
                schedule_button = gr.Button(value="Schedule", variant="primary")
                report_md = gr.Markdown(value="")

        schedule_button.click(fn=do_schedule, inputs=[prompt, steps], outputs=[report_md])

    return (demo, "Sched.", "sched"),

script_callbacks.on_ui_tabs(on_ui_tabs)
