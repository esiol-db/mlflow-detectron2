# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster Config
# MAGIC Single Node <br>
# MAGIC Databricks Runtime Version: 11.3 LTS ML (with GPU) <br>
# MAGIC Node Type: Standard_NC24s_v3 [V100] (4 GPUs)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Prep
# MAGIC If you have your own data, you can skip this!

# COMMAND ----------

# MAGIC %sh
# MAGIC # Create a directory for COCO dataset
# MAGIC mkdir -p /local_disk0/coco_dataset
# MAGIC
# MAGIC # Download train images
# MAGIC wget -O /local_disk0/coco_dataset/train2017.zip http://images.cocodataset.org/zips/train2017.zip
# MAGIC
# MAGIC # Download train images
# MAGIC wget -O /local_disk0/coco_dataset/val2017.zip http://images.cocodataset.org/zips/val2017.zip
# MAGIC
# MAGIC
# MAGIC # Download annotations
# MAGIC wget -O /local_disk0/coco_dataset/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# MAGIC
# MAGIC # Unzip the downloaded files
# MAGIC unzip -q /local_disk0/coco_dataset/train2017.zip -d /local_disk0/coco_dataset/
# MAGIC unzip -q /local_disk0/coco_dataset/val2017.zip -d /local_disk0/coco_dataset/
# MAGIC unzip -q /local_disk0/coco_dataset/annotations_trainval2017.zip -d /local_disk0/coco_dataset/
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Python main train script

# COMMAND ----------

# MAGIC %%writefile /Workspace/Users/ehsan.olfat@databricks.com/fine_tuning/Detectron/main_train.py
# MAGIC import os
# MAGIC import detectron2
# MAGIC from detectron2.engine import DefaultTrainer, default_argument_parser, launch
# MAGIC from detectron2.config import get_cfg
# MAGIC from detectron2.data.datasets import register_coco_instances
# MAGIC from detectron2.model_zoo import get_config_file, get_checkpoint_url
# MAGIC from detectron2.engine import HookBase
# MAGIC from detectron2.utils import comm
# MAGIC from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# MAGIC import mlflow
# MAGIC
# MAGIC
# MAGIC train_json_file = '/local_disk0/coco_dataset/annotations/instances_train2017.json'
# MAGIC train_image_dir = '/local_disk0/coco_dataset/train2017/'
# MAGIC
# MAGIC val_json_file = '/local_disk0/coco_dataset/annotations/instances_val2017.json'
# MAGIC val_image_dir = '/local_disk0/coco_dataset/val2017/'
# MAGIC
# MAGIC
# MAGIC # Register custom dataset in COCO format
# MAGIC register_coco_instances(
# MAGIC   name="coco_train", 
# MAGIC   metadata={}, 
# MAGIC   json_file=train_json_file, 
# MAGIC   image_root=train_image_dir
# MAGIC   )
# MAGIC register_coco_instances(
# MAGIC   name="coco_val", 
# MAGIC   metadata={}, 
# MAGIC   json_file=val_json_file, 
# MAGIC   image_root=val_image_dir
# MAGIC   )
# MAGIC
# MAGIC
# MAGIC
# MAGIC # creating a custom hook to log the total loss and the loss_rpn_cls
# MAGIC class MLflowLogger(HookBase):
# MAGIC     def __init__(self, cfg):
# MAGIC         super().__init__()
# MAGIC         self.cfg = cfg
# MAGIC         self.iter = 0
# MAGIC
# MAGIC     def after_step(self):
# MAGIC         # Log training metrics only from the main process (rank 0)
# MAGIC         if comm.get_rank() == 0:  # Check if this is the main process
# MAGIC             if self.trainer.storage.iter % 20 == 1:  # Log every 20 iterations
# MAGIC                 # Get the latest total loss
# MAGIC                 total_loss = self.trainer.storage.latest().get('total_loss')
# MAGIC                 loss_rpn_cls = self.trainer.storage.latest().get('loss_rpn_cls')
# MAGIC                 # Log the total loss to MLflow
# MAGIC                 if total_loss is not None:
# MAGIC                     mlflow.log_metric("total_loss", total_loss[0], step=self.trainer.storage.iter)
# MAGIC                     mlflow.log_metric("loss_rpn_cls", loss_rpn_cls[0], step=self.trainer.storage.iter)
# MAGIC
# MAGIC
# MAGIC     def after_epoch(self):
# MAGIC         # You can log additional epoch-level metrics here
# MAGIC         pass
# MAGIC
# MAGIC class TrainerWithMLflow(DefaultTrainer):
# MAGIC     @classmethod
# MAGIC     def build_train_loader(cls, cfg):
# MAGIC         return super().build_train_loader(cfg)
# MAGIC
# MAGIC     def build_hooks(self):
# MAGIC         # Build hooks from the parent class
# MAGIC         hooks = super().build_hooks()
# MAGIC
# MAGIC         # Add MLflow logger hook
# MAGIC         hooks.insert(0, MLflowLogger(self.cfg))
# MAGIC         
# MAGIC         return hooks
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Configuration setup
# MAGIC cfg = get_cfg()
# MAGIC cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# MAGIC cfg.DATASETS.TRAIN = ("coco_train",)
# MAGIC cfg.DATASETS.TEST = ("coco_val",)
# MAGIC cfg.DATALOADER.NUM_WORKERS = 4
# MAGIC cfg.SOLVER.BASE_LR = 0.001  # Learning rate
# MAGIC cfg.SOLVER.MAX_ITER = 100  # Number of iterations
# MAGIC cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Reduce if you run out of memory
# MAGIC cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO has 80 classes
# MAGIC cfg.OUTPUT_DIR = "/local_disk0/detectron_multigpu_output"
# MAGIC
# MAGIC # For fine-tuning, load a model pre-trained on COCO
# MAGIC cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# MAGIC
# MAGIC # Set number of GPUs
# MAGIC cfg.SOLVER.IMS_PER_BATCH = 16  # Assuming 4 GPU, this would be 4 images per batch
# MAGIC cfg.MODEL.DEVICE = "cuda"
# MAGIC
# MAGIC
# MAGIC def main():    
# MAGIC     # Log parameters at the beginning
# MAGIC     if comm.get_rank() == 0:
# MAGIC         mlflow.log_param("batch_size", cfg.SOLVER.IMS_PER_BATCH)
# MAGIC         mlflow.log_param("learning_rate", cfg.SOLVER.BASE_LR)
# MAGIC         mlflow.log_param("max_iter", cfg.SOLVER.MAX_ITER)
# MAGIC
# MAGIC
# MAGIC     trainer = TrainerWithMLflow(cfg)
# MAGIC     trainer.resume_or_load(resume=False)
# MAGIC
# MAGIC     trainer.train()
# MAGIC
# MAGIC
# MAGIC     # Log the trained model
# MAGIC     if comm.get_rank() == 0:
# MAGIC         mlflow.pytorch.log_model(trainer.model, "model")
# MAGIC
# MAGIC     # Log evaluation metrics
# MAGIC     evaluator = COCOEvaluator("coco_val", cfg, False, output_dir="/local_disk0/detectron_multigpu_output")
# MAGIC     val_loader = detectron2.data.build_detection_test_loader(cfg, "coco_val")
# MAGIC     evaluation_results = inference_on_dataset(trainer.model, val_loader, evaluator)
# MAGIC     
# MAGIC     if comm.get_rank() == 0:
# MAGIC         # Log mAP and other evaluation results only from the main process
# MAGIC         mlflow.log_metric("mAP", evaluation_results["bbox"]["AP"])
# MAGIC
# MAGIC if __name__ == "__main__":
# MAGIC     # This will automatically detect the number of available GPUs
# MAGIC     num_gpus = 4  # Change this to the number of GPUs you have
# MAGIC     # Start MLflow run
# MAGIC     
# MAGIC     launch(
# MAGIC         main,                 # The main function to run the training
# MAGIC         num_gpus_per_machine=num_gpus,  # Number of GPUs to use
# MAGIC         machine_rank=0,       # In a multi-node setting, rank of this machine (usually 0 for single machine)
# MAGIC         dist_url="tcp://127.0.0.1:{}".format(29500),  # Communication URL for distributed training
# MAGIC         args=()               # Additional arguments to pass to the training script if needed
# MAGIC     )

# COMMAND ----------

# MAGIC %md
# MAGIC # Running the python script from the notebook

# COMMAND ----------

# MAGIC %sh
# MAGIC /databricks/python/bin/pip install --upgrade pip
# MAGIC /databricks/python/bin/pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# MAGIC /databricks/python/bin/pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# MAGIC /databricks/python/bin/pip install mlflow==2.16.2
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
db_host = context.extraContext().apply('api_url')
# ""  # CHANGE THIS to your workspace URL!
db_token = context.apiToken().get()

os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
os.environ['DATABRICKS_TOKEN'] = db_token
os.environ['DATABRICKS_HOST'] = db_host
os.environ['MLFLOW_EXPERIMENT_NAME'] = "your experiment name"

# COMMAND ----------

# MAGIC %sh
# MAGIC export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_TRACKING_URI && /databricks/python/bin/python /Workspace/Users/ehsan.olfat@databricks.com/fine_tuning/Detectron/main_train.py   
