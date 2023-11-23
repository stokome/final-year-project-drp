import csv
import os
import random
import argparse
import numpy as np
from torch import nn as nn
import torch
from sklearn.preprocessing import scale, MinMaxScaler
import pandas as pd
import torch.utils.data as Data
import datetime
from models.AutoEncoder.simple_ae import Simple_Auto_Encoder
from models.AutoEncoder.deep_ae import Deep_Auto_Encoder
from models.AutoEncoder.variational_ae import Variational_Auto_Encoder

# cell
ge_folder = "./global_data/"
ge_ae_save = "./saved/Cell_line_RMA_proc_basalExp.pt"
model_save_path = "./saved/auto_encoder.pt"
device="cpu"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def train_ae(model,trainLoader,test_feature, model_type):
    start = datetime.datetime.now()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if model_type == 2:
        loss_func = loss_function
    elif model_type == 0 or model_type == 1:
        loss_func = nn.MSELoss()
    best_model=model
    best_loss=100
    for epoch in range(1, 2500 + 1 ):
        for x in trainLoader:
            y=x
            if model_type == 0 or model_type == 1:
                encoded, decoded = model(x)
                train_loss = loss_func(decoded, y)
            elif model_type == 2:
                x_hat, mean, logvar = model(x)
                train_loss = loss_func(y, x_hat, mean, logvar)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        with torch.no_grad():
            y = test_feature
            if model_type == 0 or model_type == 1:
                encoded, decoded = model(test_feature)
                test_loss = loss_func(decoded, y)
            elif model_type == 2:
                x_hat, mean, logvar = model(test_feature)
                test_loss = loss_func(y, x_hat, mean, logvar)
        if (test_loss < best_loss):
            best_loss = test_loss
            best_model = model
        if epoch%10==0:
            end = datetime.datetime.now()
            print('epoch:' ,epoch, 'train loss = ' ,train_loss.item(),"test loss:",test_loss.item(), "time:",(end - start).seconds)
    return best_model

def save_cell_oge_matrix(folder):
    f = open(folder + "Cell_line_RMA_proc_basalExp.txt")
    line = f.readline()
    elements = line.split()
    cell_names = []
    feature_names = []
    cell_dict = {}
    i = 0
    for cell in range(2, len(elements)):
        if i < 500:
            cell_name = elements[cell].replace("DATA.", "")
            cell_names.append(cell_name)
            cell_dict[cell_name] = []

    min = 0
    max = 12
    for line in f.readlines():
        elements = line.split("\t")
        if len(elements) < 2:
            print(line)
            continue
        feature_names.append(elements[1])

        for i in range(2, len(elements)):
            cell_name = cell_names[i-2]
            value = float(elements[i])
            if min == 0:
                min = value
            if value < min:
                min = value
            if max < value:
                value = max
            cell_dict[cell_name].append(value)
    #print(min)
    #print(max)
    cell_feature = []
    for cell_name in cell_names:
        for i in range(0, len(cell_dict[cell_name])):
            cell_dict[cell_name][i] = (cell_dict[cell_name][i] - min)/(max - min)
        cell_dict[cell_name] = np.asarray(cell_dict[cell_name])
        cell_feature.append(np.asarray(cell_dict[cell_name]))

    cell_feature = np.asarray(cell_feature)
    # cell_feature = cell_feature.flatten()
    # print(cell_feature.shape)
    # print((cell_feature > 11.5).sum())
    # plt.hist(cell_feature.flatten())
    # plt.show()
    # exit()
    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1

    # print(len(list(cell_dict.values())))
    # exit()
    #print(cell_dict['910927'][23])
    return cell_dict, cell_feature

lr=0.0001
batch_size=388
def main(model_type):
    random.seed(4)
    # load  gene expression data, and DNA copy number data of cell line
    cell_dict, cell_features = save_cell_oge_matrix(ge_folder)
    #normalization
    min_max = MinMaxScaler()
    cell_features = torch.tensor(min_max.fit_transform(cell_features)).float().to(device)
    ge_indim = cell_features.shape[-1]
    print(f"Cell Features: {cell_features.shape[0]}")
    # dimension reduction(gene expression data)
    if model_type == 0:
        ge_ae = Simple_Auto_Encoder(device, ge_indim, 512)
    elif model_type == 1:
        ge_ae = Deep_Auto_Encoder(device, ge_indim, 512)
    elif model_type == 2:
        ge_ae = Variational_Auto_Encoder(device, ge_indim, 512)
    train_list = random.sample((cell_features).tolist(), int(0.9 * len(cell_features)))
    test_list = [item for item in (cell_features).tolist() if item not in train_list]
    train = torch.tensor(train_list).float().to(device)
    test = torch.tensor(test_list).float().to(device)
    data_iter = Data.DataLoader(train, batch_size, shuffle=True)
    best_model = train_ae(ge_ae, data_iter, test, model_type)
    if not os.path.isdir("saved"):
        os.makedirs("./saved")
    torch.save(best_model.output(cell_features), ge_ae_save)
    torch.save(best_model, model_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train auto encoder')
    parser.add_argument('--model', type=int, required=False, default=0,     help='0: simple autoencoder, 1: deep autoencoder, 2: variational_autoencoder')
    args = parser.parse_args()
    model_type = args.model
    main(model_type)