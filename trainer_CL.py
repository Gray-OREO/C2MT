"""
PyTorch 1.11.0 implementation of the following paper:

 Usage:
    Run the trainer_CL.py:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python trainer_CL.py --exp_id=0
    ```

 Date: 2022/12/27
"""

import numpy as np
import random
from scipy import stats
from argparse import ArgumentParser
import yaml
import torch
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import time
from tensorboardX import SummaryWriter

# utils load
from dataset_update import CIQADataset, get_new_train_loaders
from models.ContraLoss import mini_ContraLoss
from models.CertaintyCoeff import mini_CertaintyCoeff
from models import build_model
from models.lookahead import Lookahead

if __name__ == "__main__":
    '''
    Args Setting for C2MT.
    '''
    parser = ArgumentParser(description='C2MT')
    parser.add_argument("--seed", type=int, default=19980427)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for data loader')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='QADS', type=str,  # Database choice
                        help='database name')
    parser.add_argument('--model', default='C2MT', type=str,
                        help='model name (default: C2MT)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    parser.add_argument('--save_path', type=str, default="results/")
    parser.add_argument('--CL', type=bool, default=True,
                        help='flag whether to disable contrastive learning strategy')
    parser.add_argument('--AL', type=bool, default=True,
                        help='flag whether to disable active learning strategy')
    parser.add_argument('--AL_start', type=int, default=10,
                        help='flag whether to disable active learning strategy')
    parser.add_argument('--AL_stop', type=int, default=30,
                        help='flag whether to disable active learning strategy')
    parser.add_argument('--AL_step', type=int, default=10,
                        help='flag whether to disable active learning strategy')
    parser.add_argument('--AL_threshold', type=int, default=0.98,
                        help='flag whether to disable active learning strategy')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config, mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    config.update(config[args.database])

    writer = SummaryWriter()
    # ======Dataloader Initialization=========
    train_dataset = CIQADataset(config, exp_id=0, status='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = CIQADataset(config, exp_id=0, status='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=args.num_workers)
    test_dataset = CIQADataset(config, exp_id=0, status='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=args.num_workers)

    if args.AL:
        certainty_coeff = torch.zeros(train_dataset.real_len + len(val_dataset) + len(test_dataset),
                                      train_dataset.patch_num, device='cuda')
        pseudo_label = torch.zeros(train_dataset.real_len + len(val_dataset) + len(test_dataset),
                                   train_dataset.patch_num, device='cuda')
    # ===========GPU Setting====================
    device = torch.device("cuda")
    # ==========Network===========
    model = build_model(config)
    model = model.to(device)
    # print(model)
    # ===========Multi-task loss function & Optimizer ==============
    criterion_0 = nn.L1Loss()
    criterion_1 = nn.CrossEntropyLoss()
    optimizer = Lookahead(Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr), la_steps=5,
                          la_alpha=1.5)
    # ===========Learning loop setting=============
    global path_checkpoint
    best_SROCC = -1
    num_p = train_dataset.patch_num
    t_epoch, t_SROCC, t_PLCC, t_KROCC, t_RMSE = 0, 0, 0, 0, 0

    for epoch in range(args.epochs):
        start_time1 = time.time()
        # train
        model.train()
        LOSS = 0
        for i, (patches, label, cls, info) in enumerate(train_loader):  # patches[b, 3, 32, 32]
            # print('Train Iter {}'.format(i))
            patches_rgb = patches.to(device)
            label = label.to(device)
            cls = cls.squeeze(-1).to(device)
            info = info.to(device)
            optimizer.zero_grad()
            outputs = model(patches_rgb)
            l1_loss = criterion_0(outputs[0], label)
            ce_loss = criterion_1(outputs[1], cls)
            if args.CL:
                contr_loss = mini_ContraLoss(outputs[2], cls)
                loss = l1_loss + ce_loss + contr_loss
                # print('  -Iter {} -Epoch {}: l1_loss:{:.4f}, ce_loss:{:.4f}, contr_loss{:.4f}'
                #       .format(i, epoch, l1_loss, ce_loss, contr_loss))
            else:
                loss = l1_loss + ce_loss
                # print('  -Iter {} -Epoch {}: l1_loss:{:.4f}, ce_loss:{:.4f}'
                #       .format(i, epoch, l1_loss, ce_loss))
            loss.backward()
            optimizer.step()
            LOSS = LOSS + loss.item()

            if args.AL:
                pred = outputs[0].detach()
                mini_CC = mini_CertaintyCoeff(pred, label.detach())
                for e in range(info.shape[0]):
                    certainty_coeff[info[e, 0], info[e, 1]] = mini_CC[e]
                    pseudo_label[info[e, 0], info[e, 1]] = pred[e]
        train_loss = LOSS / (i + 1)

        # val
        y_pred = []
        y_val = []
        model.eval()
        L = 0
        acc_cls = 0
        with torch.no_grad():
            for i, (patches, label, cls, info) in enumerate(val_loader):
                # print('Val Iter {}'.format(i))
                y_val.append(label.item())
                patches_rgb = patches.to(device)
                label = label.repeat(num_p, 1).to(device)
                cls = cls.squeeze(-1).repeat(num_p).to(device)
                outputs = model(patches_rgb)
                y_pred.append(outputs[0].mean().item())
                loss = criterion_0(outputs[0], label)
                L = L + loss.item()
                pred_cls = torch.max(outputs[1], dim=1)[1]
                acc_cls += (pred_cls == cls).sum().item()
            val_acc = acc_cls / (len(val_dataset) * num_p)
        y_val = np.array(y_val)
        y_pred = np.array(y_pred)
        val_loss = L / (i + 1)
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())

        # test
        y_pred = []
        y_test = []
        L = 0
        acc_cls = 0
        with torch.no_grad():
            for i, (patches, label, cls, info) in enumerate(test_loader):
                y_test.append(label.item())
                patches_rgb = patches.to(device)
                label = label.repeat(num_p, 1).to(device)
                cls = cls.squeeze(-1).repeat(num_p).to(device)
                outputs = model((patches_rgb))
                y_pred.append(outputs[0].mean().item())
                loss = criterion_0(outputs[0], label)
                L = L + loss.item()
                pred_cls = torch.max(outputs[1], dim=1)[1]
                acc_cls += (pred_cls == cls).sum().item()
            test_acc = acc_cls / (len(test_dataset) * num_p)
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        test_loss = L / (i + 1)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

        print("\nEpoch {} Valid Results: loss={:.4f} SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f} Acc={:.4f}"
              .format(epoch, val_loss, val_SROCC, val_PLCC, val_KROCC, val_RMSE, val_acc))

        print("         Test Results: loss={:.4f} SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f} Acc={:.4f}"
              .format(test_loss, SROCC, PLCC, KROCC, RMSE, test_acc))

        if val_SROCC > best_SROCC:
            print("Update Epoch {} as the best valid SROCC ! >>>>>>>>>>>>>>>>>>>>".format(epoch))
            best_SROCC = val_SROCC
            t_epoch, t_SROCC, t_PLCC, t_KROCC, t_RMSE = epoch, SROCC, PLCC, KROCC, RMSE

        if args.AL and args.AL_start <= (epoch + 1) <= args.AL_stop and (epoch + 1 - args.AL_start) % args.AL_step == 0:
            train_loader = get_new_train_loaders(config, pseudo_label, certainty_coeff, train_batch_size=args.batch_size,
                                                 threshold=args.AL_threshold, exp_id=0, num_workers=args.num_workers)
        end_time1 = time.time()
        Ep_time = (end_time1 - start_time1) / 60
        print('-Epoch {} takes {:.2f} mins'.format(epoch, Ep_time))
    print("\nFinal test results in epoch {}: SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}"
          .format(t_epoch, t_SROCC, t_PLCC, t_KROCC, t_RMSE))

