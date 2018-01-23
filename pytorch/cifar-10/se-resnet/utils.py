#!/usr/bin/env python
# coding: utf-8

import time
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Trainer(object):
    def __init__(self, model, optimizer, loss_func, scheduler, save_file, epoches_per_average_accuracy=10, cuda=True):
        self.model = model
        if cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_func = loss_func
        if not isinstance(scheduler, ReduceLROnPlateau):
            raise TypeError('{} is not an ReduceLROnPlateau'.format(type(scheduler).__name__))
        self.scheduler = scheduler
        self.save_file = save_file
        self.epoches_per_average_accuracy = epoches_per_average_accuracy
        self.cuda = cuda

    def _loop(self, data_loader, epoch=-1, is_train=True, show_step_time=False):
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
        mode = "train | epoch %d" % epoch if is_train else "test"
        loss = sum(losses) / len(data_loader)
        accurary = sum(accuracies) / len(data_loader.dataset)
        print("%s | loss: %.4f | accuracy: %.4f | lr: %f | using time: %.3f" % (mode, loss, accurary,
                                                                                self.optimizer.param_groups[0]['lr'],
                                                                                epoch_end_time - epoch_start_time))
        return loss, accurary

    def train(self, data_loader, epoch):
        self.model.train()
        return self._loop(data_loader, epoch, is_train=True, show_step_time=False)

    def test(self, data_loader):
        self.model.eval()
        return self._loop(data_loader, is_train=False, show_step_time=False)

    def loop(self, epochs, train_data, test_data):
        _, max_accuracy = self.test(test_data)
        train_accuracies = []

        for ep in range(epochs):
            _, train_accuracy = self.train(train_data, ep)
            train_accuracies.append(train_accuracy)
            last_average_accuracy = 0
            if len(train_accuracies) >= self.epoches_per_average_accuracy:
                train_accuracies = train_accuracies[-self.epoches_per_average_accuracy:]
                last_average_accuracy = sum(train_accuracies) / self.epoches_per_average_accuracy
            else:
                last_average_accuracy = train_accuracies[0]
            print('last average accuracy is %.4f' % last_average_accuracy)

            _, accuracy = self.test(test_data)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                self.save()

            if len(train_accuracies) == self.epoches_per_average_accuracy:
                self.scheduler.step(last_average_accuracy, epoch=ep)

    def save(self):
        print('Save model training parameters to %s' % self.save_file)
        train_state = {'model': self.model.state_dict()}
        torch.save(train_state, self.save_file)

    def load(self):
        train_state = torch.load(self.save_file)
        print('Load model training parameters from %s' % self.save_file)
        self.model.load_state_dict(train_state['model'])
