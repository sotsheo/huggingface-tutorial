from PIL import Image
import torch
from transformers import  AutoImageProcessor, AutoModelForObjectDetection

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

ckpt = 'yainage90/fashion-object-detection'
image_processor = AutoImageProcessor.from_pretrained(ckpt)
model = AutoModelForObjectDetection.from_pretrained(ckpt).to(device)

image = Image.open('https://demos.themeselection.com/sneat-bootstrap-html-admin-template/assets/img/elements/2.png').convert('RGB')

with torch.no_grad():
    inputs = image_processor(images=[image], return_tensors="pt")
    outputs = model(**inputs.to(device))
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

    items = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.item()
        label = label.item()
        box = [i.item() for i in box]
        print(f"{model.config.id2label[label]}: {round(score, 3)} at {box}")
        items.append((score, label, box))
