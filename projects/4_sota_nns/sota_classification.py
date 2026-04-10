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

# ---------------------------- classification ----------------------------
imgs = ['rhinos.jpg', 'fish.jpg', 'monkeys.jpg', 'turtles.jpg']
for img in imgs:
    # load & transform image to torch tensor
    raw_img = decode_image('../../data/images/' + img) # (channels, H, W)
    show_image(raw_img.permute(1, 2, 0).numpy(), path='../../results/classification_' + img)

    # load model
    # weights = ResNet50_Weights.DEFAULT
    # model = resnet50(weights=weights)
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights)
    model.eval()

    # prepare image
    preprocess = weights.transforms() # crop, rescale & normalization
    x = preprocess(raw_img).unsqueeze(0) # now (1, 3, s, s)
    print(x.shape)

    # prediction
    y_pred = model(x)
    prob_pred = torch.nn.functional.softmax(y_pred, dim=1).squeeze()

    top5_prob, top5_class_id = torch.topk(prob_pred, 5)
    for i in range(len(top5_prob)):
        score = top5_prob[i].item()
        category_name = weights.meta['categories'][top5_class_id[i]]

        print(f'{i + 1}: {category_name:<20} {100 * score:.2f}%')