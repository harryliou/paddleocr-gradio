import gradio as gr

def predict(img):
    return img

demo = gr.Interface(
    predict,
    inputs=[gr.Image()],
    outputs='image',
    title='PaddleOCR Demo',
    examples=[['meme.jpg']]
)

demo.launch()