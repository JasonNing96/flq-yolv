#!/bin/bash
# 8n model training
python flq_yolov_v6.py \
  --clients ./data/oil_detection_dataset/client1/oil.yaml \
            ./data/oil_detection_dataset/client2/oil.yaml \
            ./data/oil_detection_dataset/client3/oil.yaml \
            ./data/oil_detection_dataset/client4/oil.yaml \
            ./data/oil_detection_dataset/client5/oil.yaml \
            ./data/oil_detection_dataset/client6/oil.yaml \
  --val-data ./data/oil_detection_dataset/data.yaml \
  --model ./models/yolov8s.pt \
  --rounds 200 \
  --local-epochs 1 \
  --bits 8 \

# 当前状况
# v8s 效果最好，map50 ~ 0.758, 我在思考提高epoch 或者 使用yolov8m 是否可以有效提高精度和凸显低比特算法效率。 

# 8s model training
# python flq_yolov_v8.py \
#   --clients ./data/oil_detection_dataset/client1/oil.yaml \
#             ./data/oil_detection_dataset/client2/oil.yaml \
#             ./data/oil_detection_dataset/client3/oil.yaml \
#             ./data/oil_detection_dataset/client4/oil.yaml \
#             ./data/oil_detection_dataset/client5/oil.yaml \
#             ./data/oil_detection_dataset/client6/oil.yaml \
#   --val-data ./data/oil_detection_dataset/data.yaml \
#   --model ./models/yolov8s.pt \
#   --rounds 200 \
#   --local-epochs 1 \
#   --bits 8 \
#   --out-dir ./results/runs_flq_v6_yolov8s_8bit_1epochs
  
#   --out-dir ./results/runs_flq_v5
# python flq_yolo_v2.py \
#   --clients ./data/oil_detection_dataset/client1/oil.yaml \
#            ./data/oil_detection_dataset/client2/oil.yaml \
#            ./data/oil_detection_dataset/client3/oil.yaml \
#            ./data/oil_detection_dataset/client4/oil.yaml \
#            ./data/oil_detection_dataset/client5/oil.yaml \
#            ./data/oil_detection_dataset/client6/oil.yaml \
#   --val-data ./data/oil_detection_dataset/data.yaml \
#   --model ./models/yolov8n.pt \
#   --rounds 100 \
#   --local-epochs 1 \
#   --bits 8
