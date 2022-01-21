from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory


def main(opt, opt_t):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  if opt.target_dataset:
      Dataset_target = get_dataset(opt_t.target_dataset, opt_t.task)
      opt_t = opts().update_dataset_info_and_set_heads(opt_t, Dataset_target)  # target dataset
  Dataset_source = get_dataset(opt.source_dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset_source) # source dataset
  print(opt)

  logger = Logger(opt)    # record

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)  # create model
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # create optimizer
  start_epoch = 0
  if opt.load_model != '':  # load model
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task] # set trainer function
  trainer = Trainer(opt, model, optimizer)  # initial trainer
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  if opt.target_dataset:
    print('Setting up target_val data...')
    val_target_loader = torch.utils.data.DataLoader(
        Dataset_target(opt_t, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    if opt.test:
        _, preds = trainer.val(0, val_target_loader)
        val_target_loader.dataset.run_eval(preds, opt.save_dir)
  else:
    print('Setting up source_val data...')
    val_source_loader = torch.utils.data.DataLoader(
        Dataset_source(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    if opt.test:
        _, preds = trainer.val(0, val_source_loader)
        val_source_loader.dataset.run_eval(preds, opt.save_dir)
        return

    # source loader
  print('Setting up source_train data...')
  train_source_loader = torch.utils.data.DataLoader(
      Dataset_source(opt, 'train'),     # modify SOURCE dataset parameters
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  if opt.target_dataset:
        # target loader
        print('Setting up target_train data...')
        train_target_loader = torch.utils.data.DataLoader(
          Dataset_target(opt_t, 'train'),   # modify TARGET dataset parameters
          batch_size=opt_t.batch_size,
          shuffle=True,
          num_workers=opt_t.num_workers,
          pin_memory=True,
          drop_last=True
        )

        print('DA MODE')
  else:
      train_target_loader = None

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_source_loader, train_target_loader)   # do train
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items(): # log information
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),   # save last-model
                 epoch, model, optimizer)
      # with torch.no_grad():
      #   if opt.target_dataset:
      #       log_dict_val, preds = trainer.val(epoch, target_data_loader=val_target_loader)  # cal val-set loss
      #   else:
      #       log_dict_val, preds = trainer.val(epoch, source_data_loader=val_source_loader)  # cal val-set loss
      # for k, v in log_dict_val.items():
      #   logger.scalar_summary('val_{}'.format(k), v, epoch)
      #   logger.write('{} {:8f} | '.format(k, v))
      # if log_dict_val[opt.metric] < best:
      #   best = log_dict_val[opt.metric]
      #   save_model(os.path.join(opt.save_dir, 'model_best.pth'),    # save best-model
      #              epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:    # update learning rate
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),  # save lr_step-model
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt_s = opts().parse()
  opt_t = opts().parse()
  main(opt_s, opt_t)