import gradio as gr
from huggingface_hub import Repository
import os

def run_inference():
    pass

description = "迄今最好的融合换脸模型演示. \n\n" \
              "使用InsightFace的换脸和CodeFormer高分方案实现\n\n" 
examples = [["assets/rick.jpg", "assets/musk.jpg", 100, 10, []],
            ["assets/rick.jpg", "assets/rick.jpg", 100, 10, ["anonymize"]]]
article = """
Demo is based of recent research from my Ph.D work. Results expects to be published in the coming months.
"""

with gr.Blocks() as blk_demo:
    gr.Markdown(value="# 融合换脸演示")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                trg_in = gr.Image(shape=None, type="pil", label='目标人物').style(height=300)
                src_in = gr.Image(shape=None, type="pil", label='源人物').style(height=300)
            with gr.Row():
                b1 = gr.Button("换脸")
            # with gr.Row():
            #     with gr.Accordion("Options", open=False):
            #         chk_in = gr.CheckboxGroup(["Compare",
            #                                    "Anonymize",
            #                                    "Reconstruction Attack",
            #                                    "Adversarial Defense"],
            #                                   label="Mode",
            #                                   info="Anonymize mode? "
            #                                        "Apply reconstruction attack? "
            #                                        "Apply defense against reconstruction attack?")
            #         def_in = gr.Slider(0, 100, value=100,
            #                            label='Anonymization ratio (%)')
            #         mrg_in = gr.Slider(0, 100, value=100,
            #                            label='Adversarial defense ratio (%)')
            # gr.Examples(examples=[["assets/musk.jpg"], ["assets/rick.jpg"]],
                        inputs=trg_in)
        with gr.Column():
            with gr.Box():
                ano_out = gr.Image(type="pil", label='换脸结果').style(height=300)

    b1.click(run_inference, inputs=[trg_in, src_in, def_in, mrg_in, chk_in], outputs=ano_out)

    
blk_demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", server_port=6006)