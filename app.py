import gradio as gr
from huggingface_hub import Repository
import os

def run_inference():
    pass

description = "Performs subject agnostic identity transfer from a source face to all target faces. \n\n" \
              "Implementation and demo of FaceDancer, accepted to WACV 2023. \n\n" \
              "Pre-print: https://arxiv.org/abs/2210.10473 \n\n" \
              "Code: https://github.com/felixrosberg/FaceDancer \n\n" \
              "\n\n" \
              "Options:\n\n" \
              "-Compare returns the target image concatenated with the results.\n\n" \
              "-Anonymize will ignore the source image and perform an identity permutation of target faces.\n\n" \
              "-Reconstruction attack will attempt to invert the face swap or the anonymization.\n\n" \
              "-Adversarial defense will add a permutation noise that disrupts the reconstruction attack.\n\n" \
              "NOTE: There is no guarantees with the anonymization process currently.\n\n" \
              "NOTE: source image with too high resolution may not work properly!"
examples = [["assets/rick.jpg", "assets/musk.jpg", 100, 10, []],
            ["assets/rick.jpg", "assets/rick.jpg", 100, 10, ["anonymize"]]]
article = """
Demo is based of recent research from my Ph.D work. Results expects to be published in the coming months.
"""

with gr.Blocks() as blk_demo:
    gr.Markdown(value="# Face Dancer")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                trg_in = gr.Image(shape=None, type="pil", label='Target').style(height=300)
                src_in = gr.Image(shape=None, type="pil", label='Source').style(height=300)
            with gr.Row():
                b1 = gr.Button("Face Swap")
            with gr.Row():
                with gr.Accordion("Options", open=False):
                    chk_in = gr.CheckboxGroup(["Compare",
                                               "Anonymize",
                                               "Reconstruction Attack",
                                               "Adversarial Defense"],
                                              label="Mode",
                                              info="Anonymize mode? "
                                                   "Apply reconstruction attack? "
                                                   "Apply defense against reconstruction attack?")
                    def_in = gr.Slider(0, 100, value=100,
                                       label='Anonymization ratio (%)')
                    mrg_in = gr.Slider(0, 100, value=100,
                                       label='Adversarial defense ratio (%)')
            gr.Examples(examples=[["assets/musk.jpg"], ["assets/rick.jpg"]],
                        inputs=trg_in)
        with gr.Column():
            with gr.Box():
                ano_out = gr.Image(type="pil", label='Output').style(height=300)

    b1.click(run_inference, inputs=[trg_in, src_in, def_in, mrg_in, chk_in], outputs=ano_out)

"""iface = gradio.Interface(run_inference,
                         [gradio.Image(shape=None, type="pil", label='Target'),
                          gradio.Image(shape=None, type="pil", label='Source'),
                          gradio.Slider(0, 100, value=100, label="Anonymization ratio (%)"),
                          gradio.Slider(0, 100, value=100, label="Adversarial defense ratio (%)"),
                          gradio.CheckboxGroup(["compare",
                                                       "anonymize",
                                                       "reconstruction attack",
                                                       "adversarial defense"],
                                                      label='Options')],
                         "image",
                         title="Face Swap",
                         description=description,
                         examples=examples,
                         article=article,
                         layout="vertical")"""
blk_demo.launch()