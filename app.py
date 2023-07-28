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


def run_inference(source_img, target_img, face_restore, codeformer_fidelity=codeformer_fidelity, background_enhance=True, face_upsample=1,
                        target_index=-1, upscale=1, device=device, codeformer_net=codeformer_net, face_analyser=face_analyser, face_swapper=face_swapper):
    
    if codeformer_fidelity >= 1:
        codeformer_fidelity = codeformer_fidelity / 100
        
    return swap([source_img], target_img, background_enhance, codeformer_fidelity, face_upsample,
                        target_index, upscale, device, codeformer_net, face_analyser, face_swapper, face_restore)


description = "##### 迄今为止最好的开源融合换脸演示. 基于InsightFace换脸模型方案实现 \n\n" 

examples = [["./data/man1.jpeg"],
            ["./data/man2.jpeg"],
            ["./data/my_id.jpg"],
            ["./data/test_face.jpg"]
           ]

with gr.Blocks(theme=gr.themes.Glass()) as blk_demo:
    gr.Markdown(value="# 融合换脸演示")
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                trg_in = gr.Image(shape=None, type="pil", label='目标人物').style(height=300)
                gr.Examples(examples=examples,
                        inputs=trg_in)
                src_in = gr.Image(shape=None, type="pil", label='源人物').style(height=300)
                gr.Examples(examples=examples,
                        inputs=src_in)                
            with gr.Row():
                with gr.Accordion("Options", open=True):
                    face_restore = gr.Checkbox(value=True, label="高分处理")
                    # def_in = gr.Slider(0, 100, value=50,
                    #                    label='源人脸特征保持度 ratio (%)')
                    
            with gr.Row():
                b1 = gr.Button("换脸")
         
        with gr.Column():
            with gr.Box():
                ano_out = gr.Image(type="pil", label='换脸结果').style(height=300)

    b1.click(run_inference, inputs=[src_in, trg_in, face_restore, ], outputs=ano_out)

    
blk_demo.queue(concurrency_count=3).launch(server_name="0.0.0.0", server_port=6006)