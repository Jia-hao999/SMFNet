import argparse
from dataset import Dataset
from torchvision import transforms
import transform
from torch.utils import data
import torch
from collections import OrderedDict
from model.SMFNet import Model
import os
import numpy as np
import IOU
import datetime
import torch.distributed as dist
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
p = OrderedDict()
p['lr_bone'] = 1e-5  # Learning rate
p['lr_branch'] = 1e-4
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
showEvery = 200

CE = torch.nn.BCELoss(reduction='mean')
IOU = IOU.IOU(size_average=True)

def structure_loss(pred, mask):
    bce = CE(torch.nn.Sigmoid()(pred), mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce + iou


def p_structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(pred, mask)
    return bce + iou

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())

parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# train
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./checkpoints')  # 训练过程中输出的保存路径
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_thread', type=int, default=16)
parser.add_argument('--spatial_ckpt', type=str, default='./checkpoints/pretrain/spatial/spatial_bone.pth')
parser.add_argument('--flow_ckpt', type=str, default='./checkpoints/pretrain/flow/flow_bone.pth')
parser.add_argument('--depth_ckpt', type=str, default='./checkpoints/pretrain/depth/depth_bone.pth')
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

config.save_fold = config.save_fold + '/' + 'SMFNet'
if not os.path.exists("%s" % (config.save_fold)):
    os.mkdir("%s" % (config.save_fold))

if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        transform.RandomFlip(),
        transform.RandomRotate(),
        transform.colorEnhance(),
        transform.randomPeper(),
        transform.FixedResize(size=(config.input_size, config.input_size)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor()])
    dataset_train = Dataset(datasets=['DVisal','RDVS'],
                            transform=composed_transforms_ts, mode='train')

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread,
                                 drop_last=True,
                                 shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))

    net_bone = Model(3, mode=config.mode, spatial_ckpt=config.spatial_ckpt,
                     flow_ckpt=config.flow_ckpt, depth_ckpt=config.depth_ckpt)

    if config.cuda:
        net_bone = net_bone.cuda()

    base, head = [], []
    for name, param in net_bone.named_parameters():
        if 'rgb_bkbone' in name or 'flow_bkbone' in name or 'depth_bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer_bone = torch.optim.SGD([{'params': base}, {'params': head}], lr=p['lr_bone'], momentum=p['momentum'],
                                     weight_decay=p['wd'], nesterov=True)

    optimizer_bone.zero_grad()

    # 一个epoch中训练iter_num个batch
    iter_num = len(dataloader)
    for epoch in range(config.epoch):
        loss_all = 0
        loss1_all = 0
        aveGrad = 0
        optimizer_bone.param_groups[0]['lr'] = p['lr_bone']
        optimizer_bone.param_groups[1]['lr'] = p['lr_branch']
        net_bone.zero_grad()

        net_bone.train()

        for i, data_batch in enumerate(dataloader):
            image, label, flow, depth = data_batch['image'], data_batch['label'], data_batch['flow'], data_batch[
                'depth']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue
            if config.cuda:
                image, label, flow, depth = image.cuda(), label.cuda(), flow.cuda(), depth.cuda()
            decoder_out,course_flo, course_dep, sw_f = net_bone(
                image, flow, depth)

            loss1 = structure_loss(decoder_out[0], label)
            loss2 = structure_loss(decoder_out[1], label)
            loss3 = structure_loss(decoder_out[2], label)
            loss4 = structure_loss(decoder_out[3], label)
            loss5 = structure_loss(decoder_out[4], label)

            cf_loss = p_structure_loss(course_flo, label)
            cd_loss = p_structure_loss(course_dep, label)

            pf_label = (course_flo-course_dep) * label + (course_dep-course_flo) * (1-label)

            pf_label = torch.where(pf_label < 0.0, torch.tensor(0.0).cuda(), pf_label)
            pf_label = torch.where(pf_label >= 0.0, torch.tensor(1.0).cuda(), pf_label)

            swf_loss = p_structure_loss(sw_f, pf_label.detach())

            loss = loss1 + loss2 / 2 + loss3 / 4+ loss4 / 8+ loss5 / 16 + cf_loss + cd_loss + swf_loss

            optimizer_bone.zero_grad()
            loss.backward()
            optimizer_bone.step()
            loss_all += loss.data
            loss1_all += loss1.data
            if i % showEvery == 0:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f  || loss1_all : %10.4f  || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num,
                        loss1.data, loss1_all/(i + 1), loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer_bone.param_groups[0]['lr']))
            
            if i == iter_num - 1:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss ||  loss1 : %10.4f ||  loss1_all : %10.4f  || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num,
                        loss1.data,loss1_all/(i + 1),loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer_bone.param_groups[0]['lr']))

        if epoch >= 50:
            torch.save(net_bone.state_dict(),
                       '%s/epoch_%d_bone.pth' % (config.save_fold, epoch + 1))


