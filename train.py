import argparse
import os
import warnings

import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Model
from utils import train_transform, val_transform, assign_meta_id

warnings.filterwarnings("ignore")


def train(net, optim):
    net.train()
    meta_loss_data, class_loss_data, meta_true_data, class_true_data, n_data, train_progress = 0, 0, 0, 0, 0, tqdm(
        train_data_loader)
    for inputs, labels in train_progress:
        optim.zero_grad()
        out_features, out_class = net(inputs.to(device_ids[0]))
        meta_labels = meta_ids[labels]
        meta_loss = cel_criterion(out_features.permute(0, 2, 1).contiguous(), meta_labels.to(device_ids[0]))
        class_loss = cel_criterion(out_class, labels.to(device_ids[0]))
        loss = meta_loss + class_loss
        loss.backward()
        optim.step()
        meta_pred = torch.argmax(out_features, dim=-1)
        class_pred = torch.argmax(out_class, dim=-1)
        n_data += len(labels)
        meta_loss_data += meta_loss.item() * len(labels)
        class_loss_data += class_loss.item() * len(labels)
        meta_true_data += torch.sum((meta_pred.cpu() == meta_labels).float()).item() / ENSEMBLE_SIZE
        class_true_data += torch.sum((class_pred.cpu() == labels).float()).item()
        train_progress.set_description('Training Epoch {}/{} - Meta Loss:{:.4f} - Class Loss:{:.4f} - Loss:{:.4f} - '
                                       'Meta Acc:{:.2f}% - Class Acc:{:.2f}%'
                                       .format(epoch, NUM_EPOCHS, meta_loss_data / n_data, class_loss_data / n_data,
                                               (meta_loss_data + class_loss_data) / n_data, meta_true_data
                                               / n_data * 100, class_true_data / n_data * 100))

    return meta_loss_data / n_data, class_loss_data / n_data, (meta_loss_data + class_loss_data) / n_data, \
           meta_true_data / n_data * 100, class_true_data / n_data * 100


def val(net):
    net.eval()
    with torch.no_grad():
        meta_loss_data, class_loss_data, t_top1_data, t_top5_data, n_data, val_progress = 0, 0, 0, 0, 0, tqdm(
            val_data_loader)
        for inputs, labels in val_progress:
            out_features, out_class = net(inputs.to(device_ids[0]))
            meta_labels = meta_ids[labels]
            meta_loss = cel_criterion(out_features.permute(0, 2, 1).contiguous(), meta_labels.to(device_ids[0]))
            class_loss = cel_criterion(out_class, labels.to(device_ids[0]))
            n_data += len(labels)
            meta_loss_data += meta_loss.item() * len(labels)
            class_loss_data += class_loss.item() * len(labels)
            t_top1_data += torch.sum((torch.topk(out_class, k=1, dim=-1)[1].cpu() == labels.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            t_top5_data += torch.sum((torch.topk(out_class, k=5, dim=-1)[1].cpu() == labels.unsqueeze(dim=-1)).any(
                dim=-1).float()).item()
            val_progress.set_description('Val Epoch {}/{} - Meta Loss:{:.4f} - Class Loss:{:.4f} - Loss:{:.4f} - '
                                         'Acc@1:{:.2f}% - Acc@5:{:.2f}%'
                                         .format(epoch, NUM_EPOCHS, meta_loss_data / n_data, class_loss_data / n_data,
                                                 (meta_loss_data + class_loss_data) / n_data, t_top1_data /
                                                 n_data * 100, t_top5_data / n_data * 100))

    return meta_loss_data / n_data, class_loss_data / n_data, (meta_loss_data + class_loss_data) / n_data, \
           t_top1_data / n_data * 100, t_top5_data / n_data * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AntiMetric')
    parser.add_argument('--data_path', default='/home/data/imagenet/ILSVRC2012', type=str, help='path to dataset')
    parser.add_argument('--with_random', action='store_true', help='with branch random weight or not')
    parser.add_argument('--load_ids', action='store_true', help='load already generated ids or not')
    parser.add_argument('--batch_size', default=1024, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=40, type=int, help='train epoch number')
    parser.add_argument('--ensemble_size', default=12, type=int, help='ensemble model size')
    parser.add_argument('--meta_class_size', default=32, type=int, help='meta class size')
    parser.add_argument('--gpu_ids', default='0,1,2,3,4,5,6,7', type=str, help='selected gpu')

    opt = parser.parse_args()

    BATCH_SIZE, NUM_EPOCHS, LOAD_IDS, GPU_IDS = opt.batch_size, opt.num_epochs, opt.load_ids, opt.gpu_ids
    ENSEMBLE_SIZE, META_CLASS_SIZE, WITH_RANDOM = opt.ensemble_size, opt.meta_class_size, opt.with_random
    random_flag = 'random' if WITH_RANDOM else 'unrandom'
    DATA_PATH, device_ids = opt.data_path, [int(gpu) for gpu in GPU_IDS.split(',')]

    train_data_set = ImageFolder(root='{}/{}'.format(DATA_PATH, 'train'), transform=train_transform)
    train_data_loader = DataLoader(train_data_set, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_data_set = ImageFolder(root='{}/{}'.format(DATA_PATH, 'val'), transform=val_transform)
    val_data_loader = DataLoader(val_data_set, BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = DataParallel(
        Model(META_CLASS_SIZE, ENSEMBLE_SIZE, len(train_data_set.classes), WITH_RANDOM).to(device_ids[0]),
        device_ids=device_ids)
    print("# trainable parameters:", sum(param.numel() if param.requires_grad else 0 for param in model.parameters()))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(NUM_EPOCHS * 0.5), int(NUM_EPOCHS * 0.7)], gamma=0.1)
    cel_criterion = CrossEntropyLoss()

    save_name_pre = '{}_{}_{}'.format(random_flag, ENSEMBLE_SIZE, META_CLASS_SIZE)
    ids_name = 'results/{}_{}_ids.pth'.format(ENSEMBLE_SIZE, META_CLASS_SIZE)
    if LOAD_IDS:
        if os.path.exists(ids_name):
            meta_ids = torch.load(ids_name)
        else:
            raise FileNotFoundError('{} is not exist'.format(ids_name))
    else:
        meta_ids = assign_meta_id(META_CLASS_SIZE, len(train_data_set.classes), ENSEMBLE_SIZE)
        torch.save(meta_ids, ids_name)
    meta_ids = torch.tensor(meta_ids)
    results = {'train_meta_loss': [], 'train_class_loss': [], 'train_loss': [], 'train_meta_accuracy': [],
               'train_class_accuracy': [],
               'val_meta_loss': [], 'val_class_loss': [], 'val_loss': [], 'val_top1_accuracy': [],
               'val_top5_accuracy': []}

    best_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_meta_loss, train_class_loss, train_loss, train_meta_accuracy, train_class_accuracy = train(model,
                                                                                                         optimizer)
        results['train_meta_loss'].append(train_meta_loss)
        results['train_class_loss'].append(train_class_loss)
        results['train_loss'].append(train_loss)
        results['train_meta_accuracy'].append(train_meta_accuracy)
        results['train_class_accuracy'].append(train_class_accuracy)
        lr_scheduler.step(epoch)

        val_meta_loss, val_class_loss, val_loss, val_top1_accuracy, val_top5_accuracy = val(model)
        results['val_meta_loss'].append(val_meta_loss)
        results['val_class_loss'].append(val_class_loss)
        results['val_loss'].append(val_loss)
        results['val_top1_accuracy'].append(val_top1_accuracy)
        results['val_top5_accuracy'].append(val_top5_accuracy)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
        if val_top1_accuracy > best_acc:
            best_acc = val_top1_accuracy
            torch.save(model.module.state_dict(), 'epochs/{}_model.pth'.format(save_name_pre))
