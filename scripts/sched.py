from functools import partial, reduce

import torch

from modules import extra_networks, script_callbacks

from modules import prompt_parser
from modules.devices import device, dtype
from modules.sd_hijack import model_hijack

import gradio as gr
import matplotlib.pyplot as ax

ax.switch_backend("agg")

sd_model = None
sd_model_betas = None
sd_model_alphas_cumprod = None
sd_model_alphas_cumprod_prev = None


def do_restore_model_params():
    sd_model.betas = sd_model_betas
    sd_model.alphas_cumprod = sd_model_alphas_cumprod
    sd_model.alphas_cumprod_prev = sd_model_alphas_cumprod_prev

    values = sd_model_alphas_cumprod.tolist()

    x_values = list(range(sd_model.num_timesteps))

    ax.clf()  # clear current figure.
    ax.plot(x_values, values, label="original")
    ax.legend()
    ax.title("Alphas Cumulative Product")
    ax.xlabel("step")
    ax.ylabel("alphas cumprod")

    return ax


def do_update_model_params(beta_start_mil: int, beta_end_mil: int):

    torch.set_printoptions(precision=8, threshold=50)

    values = sd_model_alphas_cumprod.tolist()

    beta_start = beta_start_mil * 1.e-5
    beta_end = beta_end_mil * 1.e-5
    # beta_schedule = "linear"
    # num_train_timesteps = 1000  # default = 1000

    betas = torch.linspace(beta_start, beta_end, sd_model.num_timesteps, device=device, dtype=dtype)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        (torch.tensor([1.0], device=device, dtype=dtype), alphas_cumprod[:-1]))

    new_values = alphas_cumprod.tolist()

    x_values = list(range(sd_model.num_timesteps))

    ax.clf()  # clear current figure.
    ax.plot(x_values, values, label="original")
    ax.plot(x_values, new_values, label="update")
    ax.legend()
    ax.title("Alphas Cumulative Product")
    ax.xlabel("step")
    ax.ylabel("alphas cumprod")

    sd_model.betas = betas
    sd_model.alphas_cumprod = alphas_cumprod
    sd_model.alphas_cumprod_prev = alphas_cumprod_prev

    return ax


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


def on_model_loaded(sd_model_):
    global sd_model
    global sd_model_betas, sd_model_alphas_cumprod, sd_model_alphas_cumprod_prev

    if sd_model_ == sd_model:
        return

    sd_model = sd_model_
    sd_model_betas = sd_model_.betas.to(device, dtype)
    sd_model_alphas_cumprod = sd_model_.alphas_cumprod.to(device, dtype)
    sd_model_alphas_cumprod_prev = sd_model_.alphas_cumprod_prev.to(device, dtype)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, variant="compact") as demo:
        with gr.Row():
            with gr.Column():
                plot = gr.Plot(value=ax)
        with gr.Row():
            with gr.Column():
                beta_start = gr.Slider(minimum=5, maximum=125, step=1, label="Beta start * 1.e+5", value=85)  # 85.020
            with gr.Column():
                beta_end = gr.Slider(minimum=400, maximum=2000, step=20, label="Beta end * 1.e+5", value=1200)  # 1200.104
        with gr.Row():
            with gr.Column():
                restore_button = gr.Button(value="Restore")
            with gr.Column():
                update_button = gr.Button(value="Update", variant="primary")

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
                report_ht = gr.HighlightedText(combine_adjacent=True, adjacent_separator=' ', label="CLIP").style(color_map={'B': 'green', 'E': 'red'})
                report_md = gr.Markdown()

        restore_button.click(fn=do_restore_model_params, inputs=[], outputs=[plot])
        update_button.click(fn=do_update_model_params, inputs=[beta_start, beta_end], outputs=[plot])
        schedule_button.click(fn=do_schedule, inputs=[prompt, steps, current_step], outputs=[report_ht, report_md])

    return (demo, "Sched.", "sched"),


script_callbacks.on_model_loaded(on_model_loaded)
script_callbacks.on_ui_tabs(on_ui_tabs)
