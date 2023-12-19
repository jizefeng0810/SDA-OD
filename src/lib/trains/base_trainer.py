from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter



class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, source_batch, target_batch = None, alpha=None):
    if target_batch != None:
      source_outputs, source_grl_output, target_grl_output = self.model(source_batch['input'], x_target=target_batch['input'], alpha=alpha, DA_switch=True)  # model ouput: [hm, wh, reg]
      loss, loss_stats = self.loss(source_outputs, source_batch, source_grl_output, target_grl_output, DA_switch=True)  # calculate loss：[loss, hm_loss, wh_loss, off_loss]
      return source_outputs[-1], loss, loss_stats
    else:
      source_outputs = self.model(source_batch['input'])  # model ouput: [hm, wh, reg]
      loss, loss_stats = self.loss(source_outputs, source_batch)  # calculate loss：[loss, hm_loss, wh_loss, off_loss]
      return source_outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, source_data_loader = None, target_data_loader = None):
    if target_data_loader == None or phase == 'val':  # only source dataset
      model_with_loss = self.model_with_loss
      if phase == 'train':
        model_with_loss.train()
      else:
        if len(self.opt.gpus) > 1:
          model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

      opt = self.opt
      results = {}
      data_time, batch_time = AverageMeter(), AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
      num_iters = len(source_data_loader) if opt.num_iters < 0 else opt.num_iters
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
      end = time.time()

      for iter_id, batch in enumerate(source_data_loader):
        if iter_id >= num_iters:
          break
        data_time.update(time.time() - end)

        for k in batch:
          if k != 'meta':
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        output, loss, loss_stats = model_with_loss(batch)
        loss = loss.mean()
        if phase == 'train':
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch['input'].size(0))
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if not opt.hide_data_time:
          Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
            '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0:
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
        else:
          bar.next()

        if opt.debug > 0:
          self.debug(batch, output, iter_id)

        if opt.test:
          self.save_result(output, batch, results)
        del output, loss, loss_stats

      bar.finish()
      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
      return ret, results
    else: # source and target dataset
      model_with_loss = self.model_with_loss
      if phase == 'train':
        model_with_loss.train()
      else:
        if len(self.opt.gpus) > 1:
          model_with_loss = self.model_with_loss.module
        model_with_loss.eval()
        torch.cuda.empty_cache()

      opt = self.opt
      results = {}
      data_time, batch_time = AverageMeter(), AverageMeter()
      avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
      num_iters = len(source_data_loader) if opt.num_iters < 0 else opt.num_iters
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
      end = time.time()

      import itertools
      target_data_loader = itertools.cycle(target_data_loader)
      # for iter_id, (source_batch, target_batch) in enumerate(zip(source_data_loader, target_data_loader)):
      for iter_id, source_batch in enumerate(source_data_loader):
        target_batch = next(target_data_loader)

        p = float(((iter_id + 1) + (epoch - 1) * num_iters) / opt.num_epochs / num_iters)
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * opt.grl_weight # 0.002

        if iter_id >= num_iters:
          break
        data_time.update(time.time() - end)

        for k in source_batch:  # 挂载到cuda
          if k != 'meta':
            source_batch[k] = source_batch[k].to(device=opt.device, non_blocking=True)
            target_batch[k] = target_batch[k].to(device=opt.device, non_blocking=True)

        output, loss, loss_stats = model_with_loss(source_batch, target_batch, alpha) # Network

        for k in target_batch:  # 释放gpu
          if k != 'meta':
            target_batch[k] = target_batch[k].cpu()

        loss = loss.mean()
        if phase == 'train':
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:  # log loss information
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), source_batch['input'].size(0))
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if not opt.hide_data_time:
          Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                    '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0:
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
        else:
          bar.next()

        if opt.debug > 0:
          self.debug(source_batch, output, iter_id)

        if opt.test:
          self.save_result(output, source_batch, results)
        del output, loss, loss_stats

      bar.finish()
      ret = {k: v.avg for k, v in avg_loss_stats.items()}
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
      return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, source_data_loader=None, target_data_loader=None):
    return self.run_epoch('val', epoch, source_data_loader, target_data_loader)

  def train(self, epoch, source_data_loader, target_data_loader):
    return self.run_epoch('train', epoch, source_data_loader, target_data_loader)
