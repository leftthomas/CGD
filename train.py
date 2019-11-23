import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import DataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import Model
from utils import train_transform, val_transform, assign_meta_id


def train(net, optim):
    net.train()
    l_data, t_data, n_data, train_progress = 0, 0, 0, tqdm(train_data_loader)
    for inputs, labels in train_progress:
        optim.zero_grad()
        out = net(inputs.to(device_ids[0]))
        labels = meta_ids[labels]
        loss = cel_criterion(out.permute(0, 2, 1).contiguous(), labels.to(device_ids[0]))
        loss.backward()
        optim.step()
        pred = torch.argmax(out, dim=-1)
        n_data += len(labels)
        l_data += loss.item() * len(labels)
        t_data += torch.sum((pred.cpu() == labels).float()).item() / ENSEMBLE_SIZE
        train_progress.set_description('Epoch {}/{} - Training Loss:{:.4f} - Training Acc:{:.2f}%'
                                       .format(epoch, NUM_EPOCHS, l_data / n_data, t_data / n_data * 100))

    return l_data / n_data, t_data / n_data * 100


def val(net):
    net.eval()
    with torch.no_grad():
        l_data, t_top1_data, t_top5_data, n_data, val_progress = 0, 0, 0, 0, tqdm(val_data_loader)
        for inputs, labels in val_progress:
            out = net(inputs.to(device_ids[0]))
            meta_labels = meta_ids[labels]
            loss = cel_criterion(out.permute(0, 2, 1).contiguous(), meta_labels.to(device_ids[0]))
            n_data += len(labels)
            l_data += loss.item() * len(labels)
            out = F.normalize(out, dim=-1)
            sim_matrix = (out.cpu()[:, None, :, None, :] @ one_hot_meta_ids[None, :, :, :, None]).squeeze(
                dim=-1).squeeze(dim=-1).mean(dim=-1)
            idx = sim_matrix.argsort(dim=-1, descending=True)
            t_top1_data += (torch.sum((idx[:, 0:1] == labels.unsqueeze(dim=-1)).any(dim=-1).float())).item()
            t_top5_data += (torch.sum((idx[:, 0:5] == labels.unsqueeze(dim=-1)).any(dim=-1).float())).item()
            val_progress.set_description('Epoch {}/{} - Val Loss:{:.4f} - Val Acc@1:{:.2f}% - Val Acc@5:{:.2f}%'
                                         .format(epoch, NUM_EPOCHS, l_data / n_data, t_top1_data / n_data * 100,
                                                 t_top5_data / n_data * 100))

    return l_data / n_data, t_top1_data / n_data * 100, t_top5_data / n_data * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AntiMetric')
    parser.add_argument('--data_path', default='/home/data/imagenet/ILSVRC2012', type=str, help='path to dataset')
    parser.add_argument('--with_random', action='store_true', help='with branch random weight or not')
    parser.add_argument('--load_ids', action='store_true', help='load already generated ids or not')
    parser.add_argument('--batch_size', default=256, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
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

    model = DataParallel(Model(META_CLASS_SIZE, ENSEMBLE_SIZE, WITH_RANDOM).to(device_ids[0]), device_ids=device_ids)
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
    one_hot_meta_ids = F.one_hot(meta_ids, num_classes=META_CLASS_SIZE).float()
    results = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_top1_accuracy': [], 'val_top5_accuracy': []}

    best_acc = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        lr_scheduler.step(epoch)
        val_loss, val_top1_accuracy, val_top5_accuracy = val(model)
        results['val_loss'].append(val_loss)
        results['val_top1_accuracy'].append(val_top1_accuracy)
        results['val_top5_accuracy'].append(val_top5_accuracy)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_results.csv'.format(save_name_pre), index_label='epoch')
        if val_top1_accuracy > best_acc:
            best_acc = val_top1_accuracy
            torch.save(model.module.state_dict(), 'epochs/{}_model.pth'.format(save_name_pre))
