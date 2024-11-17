import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel

# CLIPモデルとプロセッサの読み込み
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLIPスコア計算関数
def calculate_clip_score(image, text):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    
    # ロジットから類似度スコアを計算
    logits_per_image = outputs.logits_per_image  # shape: [1, 1]
    clip_score = logits_per_image.item()  # スカラー値を取得
    return f"CLIP Score: {clip_score:.4f}"

# Gradioインターフェースの設定
with gr.Blocks() as demo:
    gr.Markdown("### CLIP Score Calculation")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Input Image")
            text_input = gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text")
            submit_button = gr.Button("Calculate CLIP Score")
        with gr.Column():
            output = gr.Textbox(label="CLIP Score")

    submit_button.click(calculate_clip_score, inputs=[image_input, text_input], outputs=output)

# アプリの起動
demo.launch(debug=True)
