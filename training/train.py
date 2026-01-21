import os
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# --- 1. REGISTRO DOS DATASETS ---
# Note que ajustamos os caminhos para a estrutura de pastas do seu repositório 
try:
    register_coco_instances("pessoas_train", {}, "data/annotations/train.coco.json", "data/images/train")
    register_coco_instances("pessoas_valid", {}, "data/annotations/valid.coco.json", "data/images/valid")
    register_coco_instances("pessoas_test", {}, "data/annotations/test.coco.json", "data/images/test")
except:
    print("Datasets já registrados.")

# --- 2. CONFIGURAÇÃO DO MODELO (FINE-TUNING) ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("pessoas_train",)
cfg.DATASETS.TEST = ("pessoas_valid",) 
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000 
cfg.SOLVER.STEPS = []

# Definindo 2 classes conforme sua correção para o Roboflow
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# --- 3. INÍCIO DO TREINAMENTO ---
trainer = DefaultTrainer(cfg) [cite: 27]
trainer.resume_or_load(resume=False)
trainer.train()
