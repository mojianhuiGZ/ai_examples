#!/usr/bin/env python
# coding: utf-8

import time
import torch
from torch.autograd import Variable


class Trainer(object):
    def __init__(self, model, optimizer, loss_func, scheduler, save_file, cuda=True):
        self.model = model
        if cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.scheduler = scheduler
        self.save_file = save_file
        self.cuda = cuda

    def _loop(self, data_loader, is_train=True, show_step_time=False):
        losses = []
        accuracies = []
        epoch_start_time = time.clock()
        for step, (data, target) in enumerate(data_loader):
            step_start_time = time.clock()
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=not is_train), Variable(target, volatile=not is_train)
            if is_train:
                self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_func(output, target)
            losses.append(loss.data[0])
            accuracies.append((output.data.max(1)[1] == target.data).sum())
            if is_train:
                loss.backward()
                self.optimizer.step()
            step_end_time = time.clock()
            if show_step_time:
                print('step %d | loss is %.4f | using time: %.3f' % (step, loss.data[0],
                                                                     step_end_time - step_start_time))
        epoch_end_time = time.clock()
        mode = "train | epoch %d" % self.scheduler.last_epoch if is_train else "test"
        loss = sum(losses) / len(data_loader)
        accurary = sum(accuracies) / len(data_loader.dataset)
        print("%s | loss: %.4f | accuracy: %.4f | lr: %f | using time: %.3f" % (mode, loss, accurary,
                                                                                self.optimizer.param_groups[0]['lr'],
                                                                                epoch_end_time - epoch_start_time))
        return loss, accurary

    def train(self, data_loader):
        self.model.train()
        return self._loop(data_loader, is_train=True, show_step_time=False)

    def test(self, data_loader):
        self.model.eval()
        return self._loop(data_loader, is_train=False, show_step_time=False)

    def loop(self, epochs, train_data, test_data):
        _, max_accuracy = self.test(test_data)
        for ep in range(epochs):
            if self.scheduler is not None:
                self.scheduler.step()
            self.train(train_data)
            _, accuracy = self.test(test_data)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.save()

    def save(self):
        print('Save model training parameters to %s' % self.save_file)
        train_state = {'model': self.model.state_dict(), 'optimizer': self.optimizer, 'scheduler': self.scheduler}
        torch.save(train_state, self.save_file)

    def load(self, last_epoch=-1):
        train_state = torch.load(self.save_file)
        print('Load model training parameters from %s' % self.save_file)
        self.optimizer = train_state['optimizer']
        self.scheduler = train_state['scheduler']
        self.model.load_state_dict(train_state['model'])
        if last_epoch > 0:
            self.scheduler.last_epoch = last_epoch
