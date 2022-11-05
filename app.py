import click
import cv2
import gradio as gr
from ocr.predict_system import TextSystem
import ocr.utility as utility
import os
from PIL import Image

res_path = 'res'

def predict(img, drop_score):
    args = utility.parse_args()
    args.drop_score = drop_score
    args.det_model_dir = os.path.join(res_path, 'ch_PP-OCRv3_det_infer')
    args.use_angle_cls = True
    args.cls_model_dir = os.path.join(res_path, 'ch_ppocr_mobile_v2.0_cls_infer')
    args.rec_model_dir = os.path.join(res_path, 'chinese_cht_PP-OCRv3_rec_infer')
    args.rec_char_dict_path = os.path.join(res_path, 'chinese_cht_dict.txt')
    args.vis_font_path = os.path.join(res_path, 'simfang.ttf')
    args.use_gpu = False

    text_sys = TextSystem(args)
    dt_boxes, rec_res, time_dict = text_sys(img)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = dt_boxes
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]
    draw_img = utility.draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores,
        drop_score=drop_score,
        font_path=args.vis_font_path
    )
    print('time spent: {}'.format(time_dict))
    return draw_img, time_dict

@click.command()
@click.option('--share', '-share', '-s', '--s', help='gradio share link', default=False)
def main(share):
    demo = gr.Interface(
        predict,
        inputs=[
            gr.Image(),
            gr.Slider(0, 1, 0.7)
        ],
        outputs=[
            gr.Image(label='Result'),
            gr.Textbox(label='Time spent')
        ],
        title='PaddleOCR Demo - https://github.com/harryliou/paddleocr-gradio',
        examples=[['meme.jpg']],
        allow_flagging='never'
    )

    demo.launch(share=share)

if __name__ == '__main__':
    main()