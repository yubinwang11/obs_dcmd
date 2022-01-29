from re import X
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import ticker
#import torchsnooper
from dataset_DCMD_evaluate import Evaluate_Dataset
import os
from pathlib import Path
from log_normalozaition import log_transfer_numpy, log_recover
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from guass_noisy import gauss_noisy

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
        run_num = max(exst_run_nums) 

curr_run = 'run%i' % run_num
run_dir = model_dir / curr_run

model = torch.load(run_dir / 'best_model.pth')

eval_dir = Path('./results_DCMD')

if not eval_dir.exists():
    eval_num = 1
else:
    exst_eval_nums = [int(str(folder.name).split('eval')[1]) for folder in
                    eval_dir.iterdir() if
                    str(folder.name).startswith('eval')]
    if len(exst_eval_nums) == 0:
        eval_num = 1
    else:
        eval_num = max(exst_eval_nums) + 1

curr_eval = 'eval%i' % eval_num
eval_dir = eval_dir / curr_eval
os.makedirs(eval_dir)



#std_x = model.std_x

model = model.cpu()
z_dim = model.z_dim

simu_len = 5000 # default=500
test_dataset = Evaluate_Dataset(simu_len=simu_len)

x_dim = test_dataset.x_dim
y_dim = test_dataset.y_dim
dt = test_dataset.dt

x_max = torch.tensor(test_dataset.x_max).to(torch.float32).numpy()
#x_max = torch.tensor(model.x_max).cpu().to(torch.float32).numpy()

X, y_seq = test_dataset.x, test_dataset.y

y_seq = torch.tensor(y_seq).unsqueeze(0).to(torch.float32)

x_hat = torch.zeros((1, simu_len, x_dim))
z_hat = torch.zeros((1, simu_len, z_dim))

with torch.no_grad():
    #z_hat[:, 0, :] = model.encode(x_hat[:, 0, :] / std_x)
    z_hat[:, 0, :] = model.encode(x_hat[:, 0, :])

for k in range(simu_len):
    if k == 0:
        z_next = z_hat[:, k, :]

    else:
        z = z_hat[:, k - 1, :]
        z_next = model.z_next_eval(z, y_seq[:, k - 1, :])

    z_hat[:, k, :] = z_next
    #x_hat[:, k, :] = model.decode(z_next) * std_x
    x_hat[:, k, :] = model.decode(z_next) 

x_hat = x_hat.squeeze(0).detach().numpy()
z_hat = z_hat.squeeze(0).detach().numpy()

x_seq = X

for k in range(simu_len):
    x_seq[k, :] = log_recover(x_seq[k, :], x_max)
    x_hat[k, :] = log_recover(x_hat[k, :], x_max)

t = [k * dt for k in range(simu_len)]

temp_state = [1, 4, 6, 7, 8 ,9]

# plots
with open(eval_dir /'data.txt', 'w') as f:  

    for i in range(x_dim):
        plt.close()
        plt.figure()
        # plot x and obs
        legend = []
        plt.plot(t, x_seq[:, i])
        legend.append('$x_{%i}$' % (i+1,))

        plt.plot(t, x_hat[:, i], '--')
        legend.append('$\hat{x}_{%i}$' % (i+1,))
        plt.legend(legend)
        plt.get_current_fig_manager().window.showMaximized()
        fig_name = 'state estimation%i' % (i+1)
        plt.xlabel("time (s)")
        for state in temp_state:
            if state == i:
                y_label = 'temperature (℃)'
                break
            else:
                y_label = 'heat transfer rate (W)'
        plt.ylabel(y_label)
        #plt.savefig(eval_dir / fig_name, dpi=500,bbox_inches = 'tight')
        plt.savefig(eval_dir / fig_name)
        #plt.show()

        real_value = x_seq[simu_len-1, i]
        estimation = x_hat[simu_len-1, i]
        relative_err = abs((real_value-estimation)/real_value)
        print('state %i real value %.2f  estimation %.2f relative error %.3f' % (i+1, real_value, estimation, relative_err),file = f)
        #f.write('state %i real value %.2f  estimation %.2f' % (i, x_seq[simu_len-1, i], x_hat[simu_len-1, i]),file = f)

plt.close()
fig, ax = plt.subplots()

axins = ax.inset_axes((0.6, 0.2, 0.4, 0.4))
legend_err = []
for i in range(x_dim):   
    rel_err = abs((x_seq[:, i]-x_hat[:, i]) / x_seq[:, i])         
    ax.plot(t, rel_err)
    axins.plot(t, rel_err)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    axins.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=0))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('relative error')
    legend_err.append('relative error on ${x}_{%i}$' % (i+1,))       
    ax.legend(legend_err, fontsize='xx-small')

axins.set_ylim(0, 0.15)
plt.get_current_fig_manager().window.showMaximized()
plt.savefig(eval_dir / 'relative error',dpi=900)
#plt.show()

