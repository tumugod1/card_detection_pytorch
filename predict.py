# predict.py
import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm

def build_eval_tfms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = timm.create_model('efficientnet_b0', pretrained=False)
        base.classifier = nn.Identity()
        self.features = base
        self.classifier = nn.Linear(1280, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x))

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt['classes']
    img_size = ckpt.get('img_size', 224)
    model = SimpleCardClassifier(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, classes, img_size

def iter_images(path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for root, _, files in os.walk(path):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                yield os.path.join(root, f)

@torch.no_grad()
def predict_image(model, classes, img_path, tfms, device):
    img = Image.open(img_path).convert('RGB')
    x = tfms(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    conf, idx = torch.max(prob, dim=0)
    return classes[idx.item()], conf.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='best_model.pth', help='checkpoint path')
    ap.add_argument('--image', help='single image path')
    ap.add_argument('--folder', help='folder of images (recursive)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes, img_size = load_model(args.ckpt, device)
    tfms = build_eval_tfms(img_size)

    paths = []
    if args.image:  paths = [args.image]
    if args.folder: paths = list(iter_images(args.folder))
    if not paths:
        print("Bir --image veya --folder veriniz.")
        return

    for p in paths:
        try:
            label, conf = predict_image(model, classes, p, tfms, device)
            print(f"{p} -> {label} ({conf*100:.1f}%)")
        except Exception as e:
            print(f"{p} -> Hata: {e}")

if __name__ == '__main__':
    main()
