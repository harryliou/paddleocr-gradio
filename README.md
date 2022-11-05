# Detect Traditional Chinese and English with PaddleOCR
Run PaddleOCR on the gradio app to detect traditional Chinese and English in images

## Setup
```
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt
```
## Usage
```
python app.py --help
```
## Run
Run gradio locally
```
  python ./app.py
```
Run and share gradio with gradio server
```
  python ./app.py -s True
```

## Reference
* [PaddleOCR github](https://github.com/PaddlePaddle/PaddleOCR)