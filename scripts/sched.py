from functools import partial, reduce

from modules import extra_networks, script_callbacks

from modules import prompt_parser
from modules.sd_hijack import model_hijack

import gradio as gr


def do_schedule(text, steps, current_step):

    #
    # update_token_counter in modules/ui.py
    #
    try:
        text, _ = extra_networks.parse_prompt(text)

        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)

    ht = []
    md = ''

    current_prompt = None
    for when, prompt in flat_prompts:
        if current_step <= when:
            current_prompt = prompt
            break

    if current_prompt is not None:
        #
        # in modules/sd_hijack_clip.py
        #
        clip = model_hijack.clip

        batch_chunks, token_count = clip.process_texts([current_prompt])

        # used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else clip.empty_chunk() for chunks in batch_chunks]
            for x in batch_chunk:
                for token in clip.tokenizer.convert_ids_to_tokens(x.tokens):
                    if token.startswith('<|'):
                        if token == '<|startoftext|>':
                            ht.append(['.', 'B'])
                        elif token == '<|endoftext|>':
                            ht.append(['.', 'E'])
                    else:
                        ht.append([token[:-4] if token.endswith('</w>') else token, None])

        md += f'{token_count} tokens at step {current_step}\n'

    for when, prompt in flat_prompts:
        md += f'### step {when}\n'
        md += prompt
        md += f'\n'

    return ht, md


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, variant="compact") as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", show_label=False, lines=3, placeholder="Prompt")
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling steps", value=20)
            with gr.Column():
                current_step = gr.Slider(minimum=1, maximum=150, step=1, label="Count tokens at this step", value=1)
        with gr.Row():
            with gr.Column():
                schedule_button = gr.Button(value="Schedule", variant="primary")
                report_ht = gr.HighlightedText().style(color_map={'B': 'green', 'E': 'red'})
                report_md = gr.Markdown()

        schedule_button.click(fn=do_schedule, inputs=[prompt, steps, current_step], outputs=[report_ht, report_md])

    return (demo, "Sched.", "sched"),

script_callbacks.on_ui_tabs(on_ui_tabs)
