import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import datetime
import matplotlib.pyplot as plt
from preprocess.Dataset_NYC import get_dataloader
from choice_model.choice_model_GPU_nyc_scalar import ChoiceModelScalar
from choice_model.choice_model_GPU_nyc_utility import ChoiceModelUtility
from tqdm import tqdm
import argparse
from utils import redirect_log_file, Timer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def prepare_dataloader(data, opt):
    """ Load data and prepare dataloader. """
    train_data_list = [value for value in data.values()]
    trainloader = get_dataloader(train_data_list, opt.batch_size, shuffle=False)

    return trainloader

def train_epoch(model, trainloader, optimizer, opt):    
    loss_list = []
    for batch in tqdm(trainloader, mininterval=2, desc='  - (Training)   ', leave=False): ## 'mininterval'表示更新进度条的最小时间间隔

        time, precinct, race, sex = map(lambda x: x.to(opt.device), batch) 
         
        """ forward """
        batch_sum_ll, matrix_H, g_res_H = model.log_likelihood(time, precinct, race, sex)
        optimizer.zero_grad()
        
        """ backward """
        loss = - batch_sum_ll
        loss.backward()
        
        """ update parameters """
        optimizer.step()

        ## constrain alpha to be greater than 1
        model.alpha.data.clamp_(min=1.00)
        
        loss_list.append(loss.item())
        
    return loss_list, matrix_H, g_res_H
        
def train(model, trainloader, num_event, optimizer, opt):
    start_time = time.time()
    sum_loss_list = []
    for epoch in range(opt.epoch):
        loss_list, matrix_H, g_res_H = train_epoch(model, trainloader, optimizer, opt)
        sum_loss = sum(loss_list) / num_event
        sum_loss_list.append(sum_loss)
        end_time = time.time()
        
        print("[Info]: Epoch {}, Loss {}, Total Time Cost {}h".format(epoch, sum_loss, (end_time - start_time)/3600))
        # scheduler.step()
    
        model_path = "choice_model_nyc_16847samples.pt"
        torch.save(model, model_path)  

        ## save g_H
        saved_matrix_H = {}
        saved_matrix_H["matrix_H"] = matrix_H
        saved_g_res_H = {}
        saved_g_res_H["g_res_H"] = g_res_H
        # np.save("saved_matrix_H_nyc_16847samples.npy", saved_matrix_H)
        # np.save("saved_g_res_H_nyc_16847samples.npy", saved_g_res_H)
        
def main():
    """ Main function. """
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", help="Number of expert.", type=int, default=2)
    parser.add_argument("-epoch", help="Maximum training epochs.", type=int, default=1000)
    parser.add_argument("-batch_size", help="Batch size.", type=int, default=256)
    parser.add_argument("-lr", help="Learning Rate.", type=float, default=1e-4)
    parser.add_argument("-interpretability", help="Interpretability mode of choice model. Possible Choice: saturated/utility/embedding/scalar.", type=str, default='utility')
    parser.add_argument("-data_path", help="Path to dataset npy file.", type=str, default="./dataset_largescale/nyc/processed_nyc/nyc_crime_dataset_16847samples_4grids_precinct.npy")
    
    opt = parser.parse_args()
    # default device is CUDA
    # opt.device = torch.device('cuda')
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ load data """
    data_ori = np.load(opt.data_path, allow_pickle=True).item()

    """ remove unclear samples (e.g. unknown sex) """
    id = 0
    data = {}
    for id_ori in list(data_ori.keys()):
        if data_ori[id_ori]['SUSP_SEX'] != 'U':
           data[id] = data_ori[id_ori]
           id += 1
    num_event = len(list(data.keys()))
    print('[Info] Event Counts: {}'.format(num_event))
    
    """ prepare dataloader """
    trainloader = prepare_dataloader(data, opt)
    
    """ prepare model """
    if opt.interpretability == 'scalar':
        choice_model = ChoiceModelScalar(data=data, H=opt.H, device=opt.device)
    elif opt.interpretability == 'utility':
        choice_model = ChoiceModelUtility(data=data, H=opt.H, device=opt.device)

    choice_model.to(opt.device)
    
    time_location_pair_counts = choice_model.M
    print('[Info] Time Location Counts: {}'.format(time_location_pair_counts))

    """ number of learnable parameters """
    num_params = sum(p.numel() for p in choice_model.parameters() if p.requires_grad)
    print('[Info] Total learnable parameters: {}'.format(num_params))

    """ optimizer and scheduler """
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, choice_model.parameters()),
    #                        opt.lr, betas=(0.9, 0.999), eps=1e-05)
    optimizer = optim.SGD(choice_model.parameters(), lr=opt.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
    
    """ train the model """
    train(choice_model, trainloader, num_event, optimizer, opt)

print("Start time is", datetime.datetime.now(), flush=1)
with Timer("Total running time") as t:
    if __name__ == '__main__':
        main()
print("Exit time is", datetime.datetime.now(), flush=1) 
