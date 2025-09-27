import time
import torch
import numpy as np
import torch.nn as nn
from utils import *

def train(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epoch, sub, init=False):
    def init_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data)

    if init:
        model.apply(init_kaiming)
    
    train_losses = []
    for epoch in range(num_epoch):
        start = time.time()
        model.train()
        train_epoch_loss = 0
        # train
        for i, batch in enumerate(train_loader):
            eeg, gsr, ppg, video, label = (var.float().to('cuda') for var in batch)
            predict, learned_correlation, gt_correlation = model(eeg, gsr, ppg, video, label)
            kl_loss = criterion[0](predict.log(), label)
            recon_loss = criterion[1](learned_correlation, gt_correlation)
            loss = kl_loss + recon_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'checkpoint/Sub_' + str(sub) + '_epoch_' + str(epoch) + '.pth')
        
        # valid
        valid_metrics = np.array([0,0,0,0,0,0]).astype(float)
        model.eval()       
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                eeg, gsr, ppg, video, label = (var.float().to('cuda') for var in batch)
                predict, learned_correlation, gt_correlation = model(eeg, gsr, ppg, video, label)
                predict, label = predict.cpu().numpy(), label.cpu().numpy()
                valid_metrics = valid_metrics + compute_distance(predict[0], label[0])

            train_epoch_loss = train_epoch_loss / len(train_loader)

            end = time.time() - start

        train_losses.append(train_epoch_loss)
        print("< Subject{} {:.0f}% {}/{} {:.3f}s >".format(sub, (epoch + 1) / num_epoch * 100, epoch + 1, num_epoch, end), end="")
        print('train_loss =', '{:.5f}'.format(train_epoch_loss), end=" ")
        print('valid_metrics =', '{}'.format(valid_metrics / len(valid_loader)))
    
    return train_losses

def test(model, test_loader, init=False):
    def init_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data)

    if init:
        model.apply(init_kaiming)

    test_metrics = np.array([0,0,0,0,0,0]).astype(float)
    metrics = np.array([0,0,0,0,0,0]).astype(float)
    model.eval()       
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            eeg, gsr, ppg, video, label = (var.float().to('cuda') for var in batch)
            predict, _, _ = model(eeg, gsr, ppg, video, label)
            predict, label = predict.cpu().numpy(), label.cpu().numpy()
            test_metrics = test_metrics + compute_distance(predict[0], label[0])

        metrics = test_metrics / len(test_loader)
    
    return metrics