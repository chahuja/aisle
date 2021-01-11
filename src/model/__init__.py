import sys
sys.path.insert(0, '..')

## Layers for different models
from .layers import *

## Models
from .speech2gesture import *
from .snl2pose import *
from .snl2pose_transformer import *
from .snl2pose_vq import *
from .snl2pose_cluster import *
from .snl2pose_cluster_soft import *
from .pos_transformer import *
from .vanilla_transformer import *
from .joint_residuals import *
from .joint import *
from .joint_late import *
from .joint_late_vq import *
from .joint_late_vq_cluster import *
from .joint_late_vq_cluster1 import *
from .joint_late_cluster_soft import *
from .joint_late_cluster_soft_style import *
from .joint_late_cluster_soft_style_disentangle import *
from .joint_late_cluster_soft_style_learn import *
from .joint_late_cluster_soft_pos import *
from .joint_late_cluster_soft_transformer import *
from .joint_late_cluster_soft_phase import *
from .transformerLM import *
#from .joint_late_cluster_soft_latent import *
from .joint_late_cycle import *
from .vanillacnn import *
from .vanillacnn_late import *
from .vanilla_cnn_cluster import *
from .nn import *
from .style_classifier import *
from .gesticulator import *

## GAN model which is a combination of Generator and Discriminator from Models
from .gan import *

## Contrastive Learning
from .contrastive_learning import *

## Trainers - BaseTrainer, Trainer, GANTrainer
from .trainer import *

'''
speech2gesture baselines
  - non-gan
  - gans

(snl2pose) - speech and language
  - non-gan
  - gans
'''
