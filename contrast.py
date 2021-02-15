from net import resnet18, resnet34, resnet50, resnet101, resnet152

import torch
import torch.nn as nn
import numpy as np
# import pandas as pd
import tqdm
import mit_utils as utils
# import analytics
import time
import os, shutil
from mail import mail_it
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import random

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-d', '--dataset', type=int)
# parser.add_argument('-g', '--gpu_id', type=str, default=0)
parser.add_argument('-F1', '--transform_function_1', type=str)
parser.add_argument('-F2', '--transform_function_2', type=str)
# parser.add_argument('-e', '--epoch', type=int, default=60)


arg = parser.parse_args()

torch.set_default_tensor_type(torch.FloatTensor)

device = "cuda"

log_dir = "logs"
model_name = 'resnet17'
model_save_dir = '%s/%s_%s' % (log_dir, model_name, time.strftime("%m%d%H%M"))

os.makedirs(model_save_dir, exist_ok=True)
log_file = "%s_%s_%s.log" % (arg.transform_function_1, arg.transform_function_2, time.strftime("%m%d%H%M"))

log_templete = {"acc": None,
                    "cm": None,
                    "f1": None,
                "per F1":None,
                "epoch":None,
                    }

data = np.load('data.npz')
orig_x = data['x']
x = np.zeros((orig_x.shape[0],3072))
x[:,36:3036] = orig_x
x = x[:,None,:]
y = data['y']


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3)
x_train = torch.tensor(x_train, dtype=torch.float).to(device)
x_test = torch.tensor(x_test, dtype=torch.float).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)
print(x_train.shape)

import torch.nn.functional as F
from transform import Transform

def save_ckpt(state, is_best, model_save_dir, message='best_w.pth'):
    current_w = os.path.join(model_save_dir, 'latest_w.pth')
    best_w = os.path.join(model_save_dir, message)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

def transform(x, mode):
    x_ = x.cpu().numpy()

    Trans = Transform()
    if mode == 'time_warp':
        pieces = random.randint(5,20)
        stretch = random.uniform(1.5,4)
        squeeze = random.uniform(0.25,0.67)
        x_ = Trans.time_warp(x_, 100, pieces, stretch, squeeze)
    elif mode == 'noise':
        factor = random.uniform(10,20)
        x_ = Trans.add_noise_with_SNR(x_,factor)
    elif mode == 'scale':
        x_ = Trans.scaled(x_,[0.3,3])
    elif mode == 'negate':
        x_ = Trans.negate(x_)
    elif mode == 'hor_flip':
        x_ = Trans.hor_filp(x_)
    elif mode == 'permute':
        pieces = random.randint(5,20)
        x_ = Trans.permute(x_,pieces)
    elif mode == 'cutout_resize':
        pieces = random.randint(5, 20)
        x_ = Trans.cutout_resize(x_, pieces)
    elif mode == 'cutout_zero':
        pieces = random.randint(5, 20)
        x_ = Trans.cutout_zero(x_, pieces)
    elif mode == 'crop_resize':
        size = random.uniform(0.25,0.75)
        x_ = Trans.crop_resize(x_, size)
    elif mode == 'move_avg':
        n = random.randint(3, 10)
        x_ = Trans.move_avg(x_,n, mode="same")
    #     to test
    elif mode == 'lowpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5,20)
        x_ = Trans.lowpass_filter(x_, order, [cutoff])
    elif mode == 'highpass':
        order = random.randint(3, 10)
        cutoff = random.uniform(5, 10)
        x_ = Trans.highpass_filter(x_, order, [cutoff])
    elif mode == 'bandpass':
        order = random.randint(3, 10)
        cutoff_l = random.uniform(1, 5)
        cutoff_h = random.uniform(20, 40)
        cutoff = [cutoff_l, cutoff_h]
        x_ = Trans.bandpass_filter(x_, order, cutoff)

    else:
        print("Error")

    x_ = x_.copy()
    x_ = x_[:,None,:]
    return x_

def comtrast_loss(x, criterion):
    LARGE_NUM = 1e9
    temperature = 0.1
    x = F.normalize(x, dim=-1)

    num = int(x.shape[0] / 2)
    hidden1, hidden2 = torch.split(x, num)


    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0,num).to('cuda')
    masks = F.one_hot(torch.arange(0,num), num).to('cuda')


    logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature
    # print(labels)
    #
    # print(torch.cat([logits_ab, logits_aa], 1).shape)

    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),
        labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),
        labels)
    loss = torch.mean(loss_a + loss_b)
    return loss, labels, logits_ab

net = resnet18(classification=False).to(device)
net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss().to(device)

batch_size = 512

optimizer = torch.optim.SGD(net.parameters(), lr=0.1 * (batch_size / 64), momentum=0.9, weight_decay=0.00001)

epochs = 70
lr_schduler = CosineAnnealingLR(optimizer, T_max=epochs - 10, eta_min=0.05)#default =0.07
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)
optimizer.zero_grad()
optimizer.step()
scheduler_warmup.step()



train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True)

target_class = ['W', 'N1', 'N2', 'N3', 'REM']



val_acc_list = []
n_train_samples = x_train.shape[0]
iter_per_epoch = n_train_samples // batch_size + 1
best_acc = -1
err = []
best_err = 1
margin = 1


for epoch in range(epochs):
    net.train()
    loss_sum = 0
    evaluation = []
    iter = 0
    with tqdm.tqdm(total=iter_per_epoch) as pbar:
        error_counter = 0

        for X, y in train_iter:
            trans = []
            for i in range(X.shape[0]):
                t1 = transform(X[i], arg.transform_function_1)
                trans.append(t1)
            for i in range(X.shape[0]):
                t2 = transform(X[i], arg.transform_function_2)
                trans.append(t2)
            trans = np.concatenate(trans)
            trans = torch.tensor(trans, dtype=torch.float, device="cuda")

            output = net(trans)

            optimizer.zero_grad()

            l, lab_con, log_con = comtrast_loss(output, criterion)
            _, log_p = torch.max(log_con.data,1)
            evaluation.append((log_p == lab_con).tolist())
            l.backward()
            optimizer.step()
            loss_sum += l
            iter += 1
            pbar.set_description("Epoch %d, loss = %.2f" % (epoch, l.data))
            pbar.update(1)
        err = l.data
    evaluation = [item for sublist in evaluation for item in sublist]


    train_acc = sum(evaluation) / len(evaluation)
    error = 1 - train_acc
    current_lr = optimizer.param_groups[0]['lr']
    print("Epoch:", epoch,"lr:", current_lr, "error:", error, " train_loss =", loss_sum.data)
    scheduler_warmup.step()
    state = {"state_dict": net.state_dict(), "epoch": epoch}
    save_ckpt(state, best_err > error, model_save_dir)
    best_err = min(best_err, error)
#=========================

net = resnet18(classification=True).to('cuda')
net = nn.DataParallel(net)
checkpoint = torch.load(os.path.join(model_save_dir,'best_w.pth'))
net.load_state_dict(checkpoint['state_dict'], strict=False)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00001)

epochs_t = 70
lr_schduler = CosineAnnealingLR(optimizer, T_max=epochs_t - 10, eta_min=0.09)#default =0.07
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_schduler)
optimizer.zero_grad()
optimizer.step()
scheduler_warmup.step()


batch_size = 256

val_acc_list = []
n_train_samples = x_train.shape[0]
iter_per_epoch = n_train_samples // batch_size + 1
best_acc = -1

for epoch in range(epochs_t):
    net.train()
    loss_sum = 0
    evaluation = []
    iter = 0
    with tqdm.tqdm(total=iter_per_epoch) as pbar:
        for X, y in train_iter:
            output = net(X)
            _, predicted = torch.max(output.data, 1)
            evaluation.append((predicted == y).tolist())
            optimizer.zero_grad()
            l = criterion(output, y)
            l.backward()
            optimizer.step()
            loss_sum += l
            iter += 1
            pbar.set_description("Epoch %d, loss = %.2f" % (epoch, l.data))
            pbar.update(1)
    evaluation = [item for sublist in evaluation for item in sublist]
    train_acc = sum(evaluation) / len(evaluation)
    current_lr = optimizer.param_groups[0]['lr']
    print("Epoch:", epoch,"lr:", current_lr," train_loss =", loss_sum.data, " train_acc =", train_acc)
    # scheduler.step()
    scheduler_warmup.step()
    val_loss = 0
    evaluation = []
    pred_v = []
    true_v = []
    with torch.no_grad():
        net.eval()
        for X, y in test_iter:
            output = net(X)
            _, predicted = torch.max(output.data, 1)
            evaluation.append((predicted == y).tolist())
            l = criterion(output, y)
            val_loss += l
            pred_v.append(predicted.tolist())
            true_v.append(y.tolist())
    evaluation = [item for sublist in evaluation for item in sublist]
    pred_v = [item for sublist in pred_v for item in sublist]
    true_v = [item for sublist in true_v for item in sublist]

    running_acc = sum(evaluation) / len(evaluation)
    val_acc_list.append(running_acc)
    print("val_loss =", val_loss, "val_acc =", running_acc)


    state = {"state_dict": net.state_dict(), "epoch": epoch}
    save_ckpt(state, best_acc < running_acc, model_save_dir, 'best_cls.pth')
    best_acc = max(best_acc, running_acc)

print("Highest acc:", max(val_acc_list))




# =========================test
model = resnet18(classification=True).to('cuda')
checkpoint = torch.load(os.path.join(model_save_dir,'best_cls.pth'))
model.load_state_dict(checkpoint['state_dict'], strict=True)
epoch_b = checkpoint['epoch']
# model.train()
model.eval()
val_loss = 0
evaluation = []
pred_v = []
true_v = []
with torch.no_grad():
    for X, y in test_iter:
        output = model(X)
        _, predicted = torch.max(output.data, 1)
        evaluation.append((predicted == y).tolist())
        l = criterion(output, y)
        val_loss += l
        pred_v.append(predicted.tolist())
        true_v.append(y.tolist())
evaluation = [item for sublist in evaluation for item in sublist]
pred_v = [item for sublist in pred_v for item in sublist]
true_v = [item for sublist in true_v for item in sublist]

highest_acc = sum(evaluation) / len(evaluation)
print("epoch=" , epoch_b, "val_acc =", highest_acc)
def calculate_all_prediction(confMatrix):
    '''
    计算总精度：对角线上所有值除以总数
    '''
    total_sum = confMatrix.sum()
    correct_sum = (np.diag(confMatrix)).sum()
    prediction = round(100 * float(correct_sum) / float(total_sum), 2)
    return prediction


def calculate_label_prediction(confMatrix, labelidx):
    '''
    计算某一个类标预测精度：该类被预测正确的数除以该类的总数
    '''
    label_total_sum = confMatrix.sum(axis=0)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    prediction = 0
    if label_total_sum != 0:
        prediction = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return prediction


def calculate_label_recall(confMatrix, labelidx):
    '''
    计算某一个类标的召回率：
    '''
    label_total_sum = confMatrix.sum(axis=1)[labelidx]
    label_correct_sum = confMatrix[labelidx][labelidx]
    recall = 0
    if label_total_sum != 0:
        recall = round(100 * float(label_correct_sum) / float(label_total_sum), 2)
    return recall


def calculate_f1(prediction, recall):
    if (prediction + recall) == 0:
        return 0
    return round(2 * prediction * recall / (prediction + recall), 2)

cm = confusion_matrix(true_v, pred_v)
f1_macro = f1_score(true_v, pred_v, average='macro')

i=0
f1 = []
for i in range(5):
    r = calculate_label_recall(cm,i)
    p = calculate_label_prediction(cm,i)
    f = calculate_f1(p,r)
    f1.append(f)


log_templete["acc"] = '{:.3%}'.format(highest_acc)
log_templete["epoch"] = epoch_b

log_templete["cm"] = str(cm)
log_templete["f1"] = str(f1_macro)
log_templete["per F1"] = str(f1)
log = log_templete
