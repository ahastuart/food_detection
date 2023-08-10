import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# mask를 표시하는 함수 정의
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# 좌표와 레이블을 표시하는 함수 정의 (단, 사용 안할 수 있음)
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# bounding box를 표시하는 함수 정의
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

image = cv2.imread('food_tray_1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

# sam 모델 인스턴스화 및 디바이스에 할당
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# sampredictor 객체 생성
predictor = SamPredictor(sam)

# 이미지 설정 (이미지를 처리하여 이미지 임베딩을 생성함,
# SamPredictor는 이 임베딩을 기억하고 후속 마스크 예측에 사용할 것이다.)
predictor.set_image(image)

# 입력 상자 정의 (알맞는 좌표값 필요)
input_boxes = torch.tensor([
    [400, 1800, 1300, 2700],
    [1850, 1900, 2800, 2875],
    [500, 1080, 900, 1500],
    [1200, 1100, 1500, 1600],
    [1680, 1100, 1900, 1600],
    [2150, 1080, 2800, 1500],
], device=predictor.device)

# 상자를 입력 프레임으로 변환하여 예측에 사용
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
# SamPredictor.predictor로 예측 수행 (predict_torch를 사용하여 여러 입력 프롬프트 생성)
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

# 이미지와 예측된 마스크, 상자를 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()