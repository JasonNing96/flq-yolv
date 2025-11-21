# python flq_yolo_v2.py \
#   --clients ./data/oil_detection_dataset/client1/oil.yaml \
#   --val-data ./data/oil_detection_dataset/data.yaml \
#   --model ./models/yolov8n.pt \
#   --rounds 2 \
#   --local-epochs 1 \
#   --bits 8 \
#   --batch 8
#    ./data/oil_detection_dataset/client2/oil.yaml \
        

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
