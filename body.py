from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# 加载预训练模型和处理器
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# 加载测试文档图片
image = Image.open("document_scan.jpg").convert("RGB")

# 文本提取
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("识别文本:", generated_text)

# 结合文本摘要模型提取主旨
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(generated_text, max_length=50, min_length=10, do_sample=False)

print("文档主旨:", summary[0]['summary_text'])
