from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.cityscapes import Cityscapes
from .dataset.foggy_cityscapes import FoggyCityscapes
from .dataset.fake_cityscapes import Fake_Cityscapes

dataset_factory = {
  'cityscapes': Cityscapes,
  'foggy_cityscapes': FoggyCityscapes,
  'fake_cityscapes': Fake_Cityscapes,
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
  
