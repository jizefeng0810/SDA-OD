from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.cityscapes import Cityscapes
from .dataset.foggy_cityscapes import FoggyCityscapes
from .dataset.kitti_2d import KITTI2D
from .dataset.fake_kitti_2d import Fake_KITTI2D
from .dataset.fake_cityscapes import Fake_Cityscapes
from .dataset.cityscapes_caronly import Cityscapes_CarOnly
from .dataset.bdd_daytime import BDD_Daytime
from .dataset.bdd_night import BDD_Night
from .dataset.fake_bdd_daytime import Fake_BDD_Daytime

dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'cityscapes': Cityscapes,
  'foggy_cityscapes': FoggyCityscapes,
  'kitti_2d': KITTI2D,
  'fake_kitti_2d': Fake_KITTI2D,
  'fake_cityscapes': Fake_Cityscapes,
  'cityscapes_car_only': Cityscapes_CarOnly,
  'bdd_daytime': BDD_Daytime,
  'bdd_night': BDD_Night,
  'fake_bdd_daytime': Fake_BDD_Daytime,
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
