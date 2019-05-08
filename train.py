import time
import copy
import torch
import torch.nn as nn
import numpy as np
# import scipy
from numpy.matlib import repmat
from loss import relation_loss

from utils import show_progressbar, cal_map_bi

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train2(model,
           dataloaders,
           device,
           dataset_sizes,
           loss_type,
           num_epochs,
           retreival=True):
    since = time.time()

    # optimizer
    # optimizer = torch.optim.LBFGS(model.parameters())

    # com_params = [model.CommonDNN.Sequential[0].weight]
    grad_params = [param for param in model.parameters()
                   if param.requires_grad]
    optimizer = torch.optim.Adam(
        grad_params,
        lr=1e-4,
        betas=(
            0.5,
            0.99),
        weight_decay=1e-4)
    # optimizer = torch.optim.SGD(grad_params, lr=0.001, momentum=0.9, weight_decay=0.001)
    # optimizer = torch.optim.ASGD(model.parameters(), lr=1e-1)

    if retreival is True:
        best_map_bi_50, best_map_bi_all = 0., 0.
        best_map_all_50, best_map_all_all = 0., 0.
    else:
        best_acc = 0.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.98)
    xmedianet_loss_history = []

    ## ======= Training  =======
    for epoch in range(num_epochs):
        # scheduler.step()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:

            num_batch = int(
                np.ceil(
                    dataset_sizes[phase] /
                    dataloaders[phase].batch_size))

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i, batch in enumerate(dataloaders[phase]):
                imgs, texts, labels = batch[0], batch[1], batch[2]
                imgs = imgs.to(device)
                texts = texts.to(device)
                labels = labels.to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # # zero the parameter gradients
                    optimizer.zero_grad()
                    img_feas, text_feas = model(imgs, texts, return_relation_score=False)

                    # loss_1 = mcml_loss_compute(img_feas, text_feas, labels)

                    relation_score = model.cal_relation_score(img_feas, text_feas)


                    ### ==== loss ==== ###
                    labelS = pair_similarity(labels, labels)
                    labelS = labelS.reshape(-1, 1)
                    # if i%100==0:
                    #     print('')
                    #     print(relation_score)
                    #     print(labelS)

                    loss = relation_loss(relation_score, labelS)

                    # if i%1000==0:
                    #     print('\nrelation_score:{}'.format(relation_score))
                    #     print('\nlabelS:{}'.format(labelS))
                    loss = loss.to(device)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                current_loss = loss.item()
                running_loss += current_loss * imgs.size(0)
                # print('{} running_loss: {:.4f}'.format(phase, loss.item() * imgs.size(0)))
                show_progressbar([i, num_batch], loss=current_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                xmedianet_loss_history.append(epoch_loss)
            import scipy.io as sio
            sio.savemat('xmedianet_loss_history.mat', {'xmedianet_loss_history': xmedianet_loss_history})
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    ## ======= Testing  =======

    img_feas_list, text_feas_list, label_list = [], [], []
    for i, batch in enumerate(dataloaders['test']):
        imgs, texts, labels = batch[0], batch[1], batch[2]
        imgs = imgs.to(device)
        texts = texts.to(device)

        img_feas, text_feas = model(imgs, texts, return_relation_score=False)
        img_feas_list.append(img_feas)
        text_feas_list.append(text_feas)
        label_list.append(labels)

    relation_score = []
    for img_feas in img_feas_list:
        relation_score_txt = []
        for text_feas in text_feas_list:
            relation_score_txtbatch = model.cal_relation_score(img_feas, text_feas)
            # relation_score_txtbatch = relation_score_txtbatch.reshape(text_feas.size(0), img_feas.size(0)).t()
            relation_score_txtbatch = relation_score_txtbatch.reshape(img_feas.size(0), text_feas.size(0))
            relation_score_txtbatch = relation_score_txtbatch.cpu().detach().numpy()
            relation_score_txt.append(relation_score_txtbatch)
        relation_score_txt = np.concatenate(relation_score_txt, 1)
        relation_score.append(relation_score_txt)
    relation_score = np.concatenate(relation_score, 0)

    labels = np.concatenate(label_list, 0)

    if retreival is True:
       cal_map_bi(-relation_score, labels)
    else:
        pass
    print()

    # import scipy.io as sio
    # sio.savemat('xmedianet_loss_history.mat', {'xmedianet_loss_history': xmedianet_loss_history})

    # print(
    #     'Best val Acc: Img: {:4f}, Txt: {:4f}'.format(
    #         best_img_acc,
    #         best_txt_acc))

    # load best model weights
    # img_model.load_state_dict(best_img_model_wts)
    # txt_model.load_state_dict(best_txt_model_wts)
    return model

# def train2(model,
#            dataloaders,
#            device,
#            dataset_sizes,
#            loss_type,
#            num_epochs,
#            retreival=True):
#     since = time.time()
#
#     # optimizer
#     # optimizer = torch.optim.LBFGS(model.parameters())
#
#     # com_params = [model.CommonDNN.Sequential[0].weight]
#     grad_params = [param for param in model.parameters()
#                    if param.requires_grad]
#     # optimizer = torch.optim.Adam(
#     #     grad_params,
#     #     lr=1e-4,
#     #     betas=(
#     #         0.5,
#     #         0.99),
#     #     weight_decay=1e-4)
#     optimizer = torch.optim.Adam(
#         grad_params,
#         lr=1e-4)
#     # optimizer = torch.optim.SGD(grad_params, lr=0.001, momentum=0.9, weight_decay=0.001)
#     # optimizer = torch.optim.ASGD(model.parameters(), lr=1e-1)
#
#     if retreival is True:
#         best_map_bi_50, best_map_bi_all = 0., 0.
#         best_map_all_50, best_map_all_all = 0., 0.
#     else:
#         best_acc = 0.
#     scheduler = torch.optim.lr_scheduler.StepLR(
#         optimizer, step_size=10000, gamma=0.98)
#     xmedianet_loss_history = []
#     for epoch in range(num_epochs):
#         # scheduler.step()
#         print('Epoch {}/{}'.format(epoch + 1, num_epochs))
#         print('-' * 10)
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'test']:
#
#             num_batch = int(
#                 np.ceil(
#                     dataset_sizes[phase] /
#                     dataloaders[phase].batch_size))
#
#             if phase == 'train':
#                 # scheduler.step()
#                 model.train()  # Set model to training mode
#
#             else:
#                 model.eval()   # Set model to evaluate mode
#
#             running_loss = 0.0
#
#             # Iterate over data.
#             for i, batch in enumerate(dataloaders[phase]):
#                 imgs, texts, labels = batch[0], batch[1], batch[2]
#                 imgs = imgs.to(device)
#                 texts = texts.to(device)
#                 labels = labels.to(device)
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # # zero the parameter gradients
#                     optimizer.zero_grad()
#                     img_feas, text_feas = model(imgs, texts, return_relation_score=False)
#
#                     # loss_1 = mcml_loss_compute(img_feas, text_feas, labels)
#
#                     relation_score = model.cal_relation_score(img_feas, text_feas)
#
#
#                     ### ==== loss ==== ###
#                     labelS = pair_similarity(labels, labels)
#                     labelS = labelS.view(-1, 1)
#                     # if i%100==0:
#                     #     print('')
#                     #     print(relation_score)
#                     #     print(labelS)
#
#                     loss = relation_loss(relation_score, labelS)
#
#                     # if i%1000==0:
#                     #     print('\nrelation_score:{}'.format(relation_score))
#                     #     print('\nlabelS:{}'.format(labelS))
#                     loss = loss.to(device)
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                 # statistics
#                 current_loss = loss.item()
#                 running_loss += current_loss * imgs.size(0)
#                 # print('{} running_loss: {:.4f}'.format(phase, loss.item() * imgs.size(0)))
#                 show_progressbar([i, num_batch], loss=current_loss)
#
#             epoch_loss = running_loss / dataset_sizes[phase]
#             xmedianet_loss_history.append(epoch_loss)
#             print('{} Loss: {:.4f}'.format(phase, epoch_loss))
#
#         if (epoch + 1) % 1 == 0 or (epoch + 1) >= 10:
#             img_feas_list, text_feas_list, label_list = [], [], []
#             for i, batch in enumerate(dataloaders['test']):
#                 imgs, texts, labels = batch[0], batch[1], batch[2]
#                 imgs = imgs.to(device)
#                 texts = texts.to(device)
#
#                 img_feas, text_feas = model(imgs, texts, return_relation_score=False)
#                 img_feas_list.append(img_feas)
#                 text_feas_list.append(text_feas)
#                 label_list.append(labels)
#
#             relation_score = []
#             for img_feas in img_feas_list:
#                 relation_score_txt = []
#                 for text_feas in text_feas_list:
#                     relation_score_txtbatch = model.cal_relation_score(img_feas, text_feas)
#                     # relation_score_txtbatch = relation_score_txtbatch.reshape(text_feas.size(0), img_feas.size(0)).t()
#                     relation_score_txtbatch = relation_score_txtbatch.view(img_feas.size(0), text_feas.size(0))
#                     relation_score_txtbatch = relation_score_txtbatch.cpu().detach().numpy()
#                     relation_score_txt.append(relation_score_txtbatch)
#                 relation_score_txt = np.concatenate(relation_score_txt, 1)
#                 relation_score.append(relation_score_txt)
#             relation_score = np.concatenate(relation_score, 0)
#
#             labels = np.concatenate(label_list, 0)
#             # labelS = pair_similarity(torch.tensor(labels), torch.tensor(labels))
#             # labelS = torch.nn.Softmax(dim=1)(labelS)
#             # labelS = labelS.numpy()
#
#             if retreival is True:
#                 best_map_bi_50, best_map_bi_all, best_epoch = \
#                     cal_map_bi(1. - relation_score, labels, best_map_bi_50, best_map_bi_all, epoch)
#                 # best_map_all_50, best_map_all_all = \
#                 #     cal_map_all(img_feas, text_feas, labels, best_map_all_50, best_map_all_all)
#             else:
#                 # best_acc = cal_knn_acc(img_feas, labels, text_feas, labels, best_acc)
#                 pass
#             print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#
#     # import scipy.io as sio
#     # sio.savemat('xmedianet_loss_history.mat', {'xmedianet_loss_history': xmedianet_loss_history})
#
#     # print(
#     #     'Best val Acc: Img: {:4f}, Txt: {:4f}'.format(
#     #         best_img_acc,
#     #         best_txt_acc))
#
#     # load best model weights
#     # img_model.load_state_dict(best_img_model_wts)
#     # txt_model.load_state_dict(best_txt_model_wts)
#     return model


def znorm(inMat):
    col = inMat.shape[0]
    row = inMat.shape[1]
    mean_val = np.mean(inMat, axis=0)
    std_val = np.std(inMat, axis=0)
    mean_val = repmat(mean_val, col, 1)
    std_val = repmat(std_val, col, 1)
    x = np.argwhere(std_val == 0)
    for y in x:
        std_val[y[0], y[1]] = 1
    return (inMat - mean_val) / std_val

def pair_similarity(x, y):
    '''
    x: n * dx
    y: m * dy
    '''

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    ps = torch.eq(x,y).squeeze(2)
    ps = ps.float()
    # ps -= (ps == 0.).float()
    return ps