import argparse
import torch
import numpy as np
#from scipy import io
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

from NN_models_DCMD import KKL_Autoencoder
from dataset_DCMD import DCMD_Dataset

from pathlib import Path
import os

def run(config):

    if config.use_wandb==True:
        import wandb
        wandb.init(project="obs_dcmd")

    LAMBDAS = [-0.5, -1, -1.5, -0.5, -1, -1.5, -0.5, -1, -1.5, -0.5, -1, -1.5, -0.5, -1, -1.5] 
    BATCH_SIZE = 100
    NET_ARCH = [3000, 3000]

    N_SAMPLES_TRAIN = config.samples_train
    N_SAMPLES_VALID = config.samples_valid
    CRITERION = torch.nn.MSELoss()

    # %% --- init datasets ---

    train_dataset = DCMD_Dataset(n_samples=N_SAMPLES_TRAIN, cuda_button=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # default shuffle=True 

    std_x = train_dataset.std_x
    x_max = train_dataset.x_max

    valid_dataset = DCMD_Dataset(n_samples=N_SAMPLES_VALID, cuda_button=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True) # default shuffle=True 

    x_dim = train_dataset.x_dim
    y_dim = train_dataset.y_dim
    dt = train_dataset.dt

    # %% --- init model ---
    model = KKL_Autoencoder(x_dim=x_dim,
                            y_dim=y_dim,
                            dt=dt,
                            lambdas=LAMBDAS,
                            net_arch=NET_ARCH,model_togpu=True)
    
    model.std_x = std_x
    model.x_max = x_max

    # %% --- training T ---
    LEARNING_RATE = config.learning_rate
    optimizer_T = torch.optim.Adam(model.T.parameters(), lr=LEARNING_RATE)

    if torch.cuda.is_available():
        model = model.cuda()

    print("==== training T (encoder) ====")
    best_loss = np.inf
    for epoch in range(config.train_epoch):  
        train_losses = []
          
        if epoch < config.exp_epoch:
            # scale z thanks to B
            loader = DataLoader(train_dataset, batch_size=len(train_dataset))
            scaling_data = list(loader)[0]

            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()

            with torch.no_grad():

                x = scaling_data[0]
                z = model.encode(x)
                scale = 1 / z.std(dim=0)
                                                  
                rel_scale = scale / scale.mean()

                for i, s in enumerate(rel_scale):
                    model.B.data[i] *= s
                print('\t B : ', *model.B.detach())
               
        for x, y, x_next, y_next in tqdm(train_loader):

            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()

            optimizer_T.zero_grad()

            z = model.encode(x)

            z_dyn = model.z_next(z, y)
            z_next = model.encode(x_next)

            loss_T = CRITERION(z_next, z_dyn)

            if config.use_wandb==True:
                wandb.log({'epoch': epoch, 'loss_T': loss_T})

            loss_T.backward()
            optimizer_T.step()
            train_losses.append(loss_T.item())

        with torch.no_grad():

            for i, (x, y, x_next, y_next) in tqdm(enumerate(valid_loader)):
                z = model.encode(x)
                z_dyn = model.z_next(z, y)
                valid_loss_T = CRITERION(model.encode(x_next), z_dyn).item()

            if valid_loss_T < best_loss:
                best_loss = valid_loss_T
                best_model = copy.deepcopy(model)

        print('epoch %i train %.2e    valid %.2e' % (epoch, np.mean(train_losses),
                                                    valid_loss_T))

        if epoch > 0 and epoch % 50 == 0:  
            optimizer_T.param_groups[0]['lr'] /= 4
            print('reduce optim lr = %.2e' % (optimizer_T.param_groups[0]['lr'],))

    model = copy.deepcopy(best_model)

    # %% --- training T⁻1 ---
    LEARNING_RATE = config.learning_rate
    optimizer_invT = torch.optim.Adam(model.Psi.parameters(), lr=LEARNING_RATE)

    print("==== training pseudo inverse (decoder) ====")
    best_loss = np.inf
    for epoch_inv in range(config.train_epoch): 
        train_losses = []

        for x, y, x_next, y_next in tqdm(train_loader):

            if hasattr(torch.cuda, 'empty_cache'):
	            torch.cuda.empty_cache()

            with torch.no_grad():
                z_ = model.encode(x)
                z_next = model.encode(x_next)
                
            optimizer_invT.zero_grad()
            
            x_hat = model.decode(z)
            x_next_hat = model.decode(z_next)

            if config.data_normalization==True:
                loss_invT = CRITERION(x, x_hat) + CRITERION(x_next, x_next_hat) 

                if config.use_wandb==True:
                    wandb.log({'epoch_inv': epoch_inv, 'losss_invT': loss_invT})

            else:
                
                loss_invT_Flowrate = CRITERION(x[:,0], x_hat[:,0]) + CRITERION(x[:,2], x_hat[:,2]) + CRITERION(x[:,3], x_hat[:,3]) + CRITERION(x[:,5], x_hat[:,5])
                loss_invT_Temp = CRITERION(x[:,1], x_hat[:,1]) + CRITERION(x[:,4], x_hat[:,4]) + CRITERION(x[:,6], x_hat[:,6]) + CRITERION(x[:,7], x_hat[:,7]) + CRITERION(x[:,8], x_hat[:,8]) + CRITERION(x[:,9], x_hat[:,9])
                loss_invT =  loss_invT_Flowrate / int(1e6) + loss_invT_Temp / int(1e2)
                
                if config.use_wandb==True:
                    wandb.log({'epoch_inv': epoch_inv, 'loss_invT': loss_invT, 'loss_invT_Flowrate': loss_invT_Flowrate, 'loss_invT_Temp': loss_invT_Temp})

            loss_invT.backward()
            optimizer_invT.step()

            train_losses.append(loss_invT.item())

        with torch.no_grad():

            for i, (x, y, x_next, y_next) in tqdm(enumerate(valid_loader)):
                z = model.encode(x)
                valid_loss_invT = CRITERION(x, model.decode(z.detach())).item()

            if valid_loss_invT < best_loss:
                best_loss = valid_loss_invT
                best_model = copy.deepcopy(model)

        print('epoch_inv %i train %.2e    valid %.2e' % (epoch_inv, np.mean(train_losses),
                                                    valid_loss_invT))

        if epoch_inv > 0 and epoch_inv % 50 == 0: ##default 50
            optimizer_invT.param_groups[0]['lr'] /= 4
            print('reduce optim lr = %.2e' % (optimizer_invT.param_groups[0]['lr'],))

    if config.save_model==True:

        model_dir = Path('./models')
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                            model_dir.iterdir() if
                            str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                run_num = 1
            else:
                run_num = max(exst_run_nums) + 1 

        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run

        os.makedirs(run_dir)
        torch.save(best_model, run_dir / 'best_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-epoch", default=160, type=int) 
    parser.add_argument("--exp-epoch", default=30, type=int)        
    parser.add_argument("--samples-train", default=200 * int(1e3), type=int)
    parser.add_argument("--samples-valid", default=50 * int(1e3), type=int)
    '''
    parser.add_argument("--samples-train", default=20000, type=int)
    parser.add_argument("--samples-valid", default=5000, type=int)
    '''
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--use-wandb", default=False, type=bool)
    parser.add_argument("--save-model", default=False, type=bool)
    parser.add_argument("--data-normalization", default=True, type=bool)  

    config = parser.parse_args()

    run(config)