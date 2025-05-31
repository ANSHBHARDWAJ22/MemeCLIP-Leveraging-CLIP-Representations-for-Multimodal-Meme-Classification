import os
from yacs.config import CfgNode 

cfg = CfgNode()

# ✅ Root directories
cfg.root_dir = "/content/memeCLIP_PROJECT"
cfg.img_folder = "/content/memeCLIP_PROJECT/Images"
cfg.info_file = "/content/memeCLIP_PROJECT/PrideMM.csv"

# ✅ Checkpoint paths
cfg.checkpoint_path = "/content/drive/MyDrive"
cfg.checkpoint_file = "/content/drive/MyDrive/model.ckpt"

# ✅ CLIP and dataset setup
cfg.clip_variant = "ViT-L/14"
cfg.dataset_name = 'PrideMM'
cfg.name = 'MemeCLIP' 
cfg.label = 'hate'
cfg.seed = 42
cfg.test_only = False
cfg.device = 'cuda'
cfg.gpus = [0]
cfg.reproduce = True  # Ensures deterministic results

# ✅ Label-specific class names
if cfg.label == 'hate':
    cfg.class_names = ['Benign Meme', 'Harmful Meme']
elif cfg.label == 'humour':
    cfg.class_names = ['No Humour', 'Humour']
elif cfg.label == 'target':
    cfg.class_names = ['No particular target', 'Individual', 'Community', 'Organization']
elif cfg.label == 'stance':
    cfg.class_names = ['Neutral', 'Support', 'Oppose']

# ✅ Model training hyperparameters
cfg.batch_size = 16
cfg.image_size = 224
cfg.num_mapping_layers = 1
cfg.unmapped_dim = 768
cfg.map_dim = 1024
cfg.num_pre_output_layers = 1
cfg.drop_probs = [0.1, 0.4, 0.2]
cfg.lr = 1e-4
cfg.max_epochs = 10
cfg.ratio = 0.2
cfg.weight_decay = 1e-4
cfg.num_classes = len(cfg.class_names)
cfg.scale = 30
cfg.print_model = True
