import torch
from torchvision.io.image import decode_image
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    ResNet50_Weights,
    efficientnet_v2_l,
    resnet50,
)
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

# from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)
from torchvision.utils import draw_bounding_boxes

torch.backends.cudnn.deterministic = True
import matplotlib.pyplot as plt
import numpy as np

from postprocessing import show_image

# ----------------------------- segmentation -----------------------------
# load & transform image to torch tensor
raw_img = decode_image("../../data/images/goose.jpg")  # (channels, H, W)
# raw_img = decode_image('../../data/images/courtyard.jpg') # (channels, H, W)
img_max, img_min = raw_img.max(), raw_img.min()
show_image(raw_img.permute(1, 2, 0).numpy(), path="../../results/segmentation1.jpg")

# load model
# weights = DeepLabV3_ResNet50_Weights.DEFAULT
# model = deeplabv3_resnet50(weights=weights)
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# prepare image
preprocess = weights.transforms()  # center crop, rescale & normalization
x = preprocess(raw_img).unsqueeze(0)
print(x.shape)

# predict
y_pred = model(x)["out"]  # 'aux' not needed during inference

normalized_masks = y_pred.softmax(dim=1)
class_map = normalized_masks.argmax(dim=1).squeeze()
unique_class_ids = torch.unique(class_map)

# ---------------------------- postprocessing ----------------------------
input_img = (x - x.min()) / (x.max() - x.min())
input_img = (input_img * (img_max - img_min) + img_min).to(
    torch.uint8
)  # find max value above + min
input_img = input_img.squeeze().permute(1, 2, 0).numpy()

colors = ["Oranges_r", "Blues_r", "Reds_r", "Purples_r", "Greens_r"]
W, H = input_img.shape[0], input_img.shape[1]
plt.figure(figsize=(H / 100, W / 100), dpi=100)
plt.imshow(input_img)
for i, class_id in enumerate(unique_class_ids):
    mask = class_map == class_id
    name = weights.meta["categories"][class_id]
    print(f'class "{name}" coverS {mask.sum()} pixels')

    mask = np.ma.masked_where(class_map == False, mask)
    plt.imshow(mask, alpha=0.5, cmap=colors[i - 1])
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../../results/segmentation2.jpg")
plt.show()
