# training and evaluation loop functions

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.utils import save_image
from .vgg import Vgg16


def train(model, train_dataloader, val_dataloader, log_dir, ckpt_path, device,
          n_epochs=1000, eval_every_nth_epoch=50, sched_patience=5, init_lr=1.0e-4):

    vgg = Vgg16().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       patience=sched_patience,
                                                       verbose=True)

    for epoch_id in tqdm(range(0, n_epochs)):

        model.train()
        torch.save(model.state_dict(), ckpt_path)

        for batch_idx, (x, ytrue, img_names) in enumerate(train_dataloader):
            x, ytrue = x.to(device), ytrue.to(device)
            ypred = model(x)
            vgg_loss = criterion_l1(vgg(ypred), vgg(ytrue))
            optimizer.zero_grad()
            vgg_loss.backward()
            optimizer.step()

        if epoch_id % eval_every_nth_epoch == 0:
            eval_dir = os.path.join(log_dir, 'val_preds_%04d' % epoch_id)
            val_loss = evaluate(model, val_dataloader, eval_dir, device, vgg)
            sched.step(val_loss)

    return


def evaluate(model, data_loader, res_dir, device, vgg=None, report_loss=True):

    model.eval()

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if vgg is None:
        vgg = Vgg16().to(device)
    criterion_l1 = nn.L1Loss().to(device)
    losses = []

    for batch_idx, (x, ytrue, img_names) in tqdm(enumerate(data_loader)):

        x, ytrue = x.to(device), ytrue.to(device)

        ypred = model(x).detach().to(device)
        losses.append(float(criterion_l1(vgg(ypred), vgg(ytrue))))

        for fid in range(0, len(img_names)):
            save_image(ypred[fid].transpose(1, 2), os.path.join(res_dir, '%s' % img_names[fid]))

    avg_loss = np.mean(losses)

    if report_loss:
        print("\nmean VGG loss: %f" % np.mean(avg_loss))
    print("images saved at %s" % res_dir)

    return avg_loss
