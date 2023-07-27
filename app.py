import gradio as gr
from huggingface_hub import Repository
import os
import cv2
import copy
import argparse
import insightface
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
from restoration import *
from swapper import swap


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints")
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


# make sure the ckpts downloaded successfully
check_ckpts()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                 codebook_size=1024,
                                                 n_head=8,
                                                 n_layers=9,
                                                 connect_list=["32", "64", "128", "256"],
                                                 ).to(device)
ckpt_path = "/data/CodeFormer/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

model = "./checkpoints/inswapper_128.onnx"
# load face_analyser
face_analyser = getFaceAnalyser(model)
# load face_swapper
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
face_swapper = getFaceSwapModel(model_path)
codeformer_fidelity = 0.5


def run_inference(source_img, target_img, background_enhance=True, codeformer_fidelity=codeformer_fidelity, face_upsample=1,
                        target_index=-1, upscale=1, device=device, codeformer_net=codeformer_net, face_analyser=face_analyser, face_swapper=face_swapper):
    return swap([source_img], target_img, background_enhance, codeformer_fidelity, face_upsample,
                        target_index, upscale, device, codeformer_net, face_analyser, face_swapper)



description = "迄今最好的融合换脸模型演示. \n\n" \
              "使用InsightFace的换脸和CodeFormer高分方案实现\n\n"
examples = [["/data/man1.jpg", "assets/musk.jpg", 100, 10],
            ["/data/man1.jpg", "assets/rick.jpg", 100, 10]]
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
            with gr.Row():
                with gr.Accordion("Options", open=False):
                    # chk_in = gr.CheckboxGroup(["Compare",
                    #                            "Anonymize",
                    #                            "Reconstruction Attack",
                    #                            "Adversarial Defense"],
                    #                           label="Mode",
                    #                           info="Anonymize mode? "
                    #                                "Apply reconstruction attack? "
                    #                                "Apply defense against reconstruction attack?")
                    def_in = gr.Slider(0, 100, value=100,
                                       label='Anonymization ratio (%)')
                    mrg_in = gr.Slider(0, 100, value=100,
                                       label='Adversarial defense ratio (%)')
            # gr.Examples(examples=[["assets/musk.jpg"], ["assets/rick.jpg"]],
            #             inputs=trg_in)
        with gr.Column():
            with gr.Box():
                ano_out = gr.Image(type="pil", label='换脸结果').style(height=300)

    b1.click(run_inference, inputs=[src_in, trg_in], outputs=ano_out)

    
blk_demo.queue(concurrency_count=3).launch(server_name="0.0.0.0", server_port=6006)