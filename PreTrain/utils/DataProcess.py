from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
import pandas as pd
import os
import numpy as np


def to_label(description):
    label = np.zeros(14)
    description_lis = description.split('_')

    if 'Drive' in description_lis: label[0] = 1
    if 'Fan' in description_lis: label[1] = 1
    if 'normal' in description_lis: label[2] = 1
    if '48k' in description_lis: label[3] = 1

    if '0' in description_lis: label[13] = 0
    if '1' in description_lis: label[13] = 1
    if '2' in description_lis: label[13] = 2
    if '3' in description_lis: label[13] = 3

    if 'End' in description_lis:
        description_part1 = description_lis[3].split('@')
        if '3' in description_part1: label[10] = 1
        if '6' in description_part1: label[11] = 1
        if '12' in description_part1: label[12] = 1

        description_part2 = description_part1[0].split('0')
        if description_part2[-1] == '7': label[8] = 1
        if description_part2[-1] == '14': label[8] = 2
        if description_part2[-1] == '21': label[8] = 3
        if description_part2[-1] == '28': label[9] = 4

        if description_part2[0] == 'B': label[4] = 1
        if description_part2[0] == 'IR': label[5] = 1
        if description_part2[0] == 'OR': label[6] = 1
        if description_part2[0] == 'nor': label[7] = 1

    return label

    # 存入label.csv
    # import pandas as pd
    # # 准备数据
    # data_df = pd.DataFrame(lit)  # 关键1，将ndarray格式转换为DataFrame
    # # 将文件写入excel表格中
    # writer = pd.ExcelWriter('hhh.xlsx')  # 关键2，创建名称为hhh的excel表格
    # data_df.to_excel(writer, 'page_1',
    #                  float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    # writer.save()  # 关键4


class signal_caption_dataset(Dataset):
    def __init__(self, df):
        self.signals = df["signal"]
        self.caption = df["caption"]

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        # signals = self.preprocess(self.signals[idx])
#       # signals = sig_preprocess(np.loadtxt(self.signals[idx]))
        signals = pd.read_csv(self.signals[idx], header=None)
        signals = sig_preprocess(signals[0])
        caption = torch.Tensor(self.caption[idx])
        # caption = caption.long()  # 不是转long，是向下取整
        caption = caption.to(torch.int64)  # 不是转long，是向下取整
        return signals, caption


def load_data(signal_path, batch_size):
    df = {'signal': [], 'caption': []}
    img_descriptions = os.listdir(signal_path)

    for description in img_descriptions:
        description_path = signal_path + '/' + description
        # cup_list = os.listdir(cup_path)
        # cupnot_list = os.listdir(cupnot_path)

        description_list = os.listdir(description_path)
        caption = description_path.split('/')[-1]
        caption = to_label(caption)
        for sig in description_list:
            sig_path = description_path + '/' + sig
            df['signal'].append(sig_path)
            df['caption'].append(caption)

    dataset = signal_caption_dataset(df)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def sig_preprocess(signal):
    signal = torch.Tensor(signal)
    signal = signal.reshape(2048)
    nn.functional.normalize(signal, dim=0)
    signal = signal.unsqueeze(0)  # 行不通
    return signal