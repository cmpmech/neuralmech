from torchvision.io.image import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
# from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import torch
torch.backends.cudnn.deterministic = True
import numpy as np
from postprocessing import show_image

# --------------------------- object detection ---------------------------
# load & transform image to torch tensor
raw_img = decode_image('../../data/images/gooses.jpg') # (channels, H, W)
# raw_img = decode_image('../../data/images/firebrigade.jpg') # (channels, H, W)
img_max, img_min = raw_img.max(), raw_img.min()
show_image(raw_img.permute(1, 2, 0).numpy(), path='../../results/objectdetection1.jpg')

# load model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.8)
model.eval()

# prepare image
preprocess = weights.transforms() # center crop, rescale & normalization
x = preprocess(raw_img).unsqueeze(0)
print(x.shape)

# predict
y_pred = model(x)[0] # list of dicts (one for each image in batch)

# filter results
keep = y_pred['scores'] > 0.8
boxes = y_pred['boxes'][keep]
labels = y_pred['labels'][keep]
scores = y_pred['scores'][keep]

pred_labels = []
for i in range(len(boxes)):
    score = scores[i].item()
    label_id = labels[i].item()
    label_name = weights.meta["categories"][label_id]
    pred_labels.append(f'{label_name}: {100 * score:.1f}%')
    print(f'{i+1}: {label_name:<20} {100 * score:.1f}%')

# ---------------------------- postprocessing ----------------------------
norm_img = x[0]
min_val, max_val = norm_img.min(), norm_img.max()
img_uint8 = ((norm_img - min_val) / (max_val - min_val) * (img_max - img_min) + img_min).to(torch.uint8)

box_img = draw_bounding_boxes(
    img_uint8,
    boxes=boxes,
    labels=pred_labels,
    colors="white",
    font="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    width=4,
    font_size=44
)

img_np = box_img.permute(1, 2, 0).numpy() # (C,H,W) -> (H,W,C)
show_image(img_np, path='../../results/objectdetection2.jpg')