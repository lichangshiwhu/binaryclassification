import os
import random

import torch

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import MinMaxScaler

from model import getModel

inputSize = 2
outputSize = 1

step = 0.001

# sigmoid gradient
def getSigmoidGrad(t, Y):
    bottom = np.exp(-2*Y*t) + np.exp(2*Y*t) + 2
    return -4*Y / bottom
# logistic gradient
def getLogisticGrad(t, Y):
    return -Y / (np.exp(Y*t) + 1)

x = np.arange(0, 1.1, step)
y = np.arange(0, 1.1, step)

X,Y = np.meshgrid(x,y)
shape0,shape1 = X.shape

print(X.shape)
print(Y.shape)

X1 = np.expand_dims(X, axis = 2)
Y1 = np.expand_dims(Y, axis = 2)

data = np.concatenate((X1, Y1), axis=2).reshape(-1, 2)
data = torch.utils.data.TensorDataset(torch.FloatTensor(data))

test_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

def getdatasets(noise):
    Xm, Ym = make_moons(n_samples = 1000, noise = noise)

    scale = MinMaxScaler()
    Xm = scale.fit_transform(Xm)

    x1 = np.array([[1, 0.05],[0.99, 0.04]])
    y1 = np.array([0, 0])
    Xm = np.concatenate((x1, Xm))
    Ym = np.concatenate((y1, Ym))

    # Xmp = Xm[Ym==1]
    # Xmn = Xm[Ym==0]
    return Xm, Ym

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) 

def getmodeloutput(pthfile, width, depth):
    modelClass = getModel('DNN')
    model = modelClass(inputSize=inputSize, width= width, deep= depth, outputSize = outputSize)
    model.load_state_dict(torch.load(pthfile))
    model.eval()

    output = []
    for _, (test_data) in enumerate(test_loader):
        subOutput = model(test_data[0])
        output.append(subOutput.detach().numpy())

    output = np.concatenate(output, axis=0)

    output = output.reshape(shape0, shape1)
    return output

pthfiles = [[ './savemodels/LogisticLoss.pth', './savemodels/SigmoidLoss.pth']]
# ['./savemodels/SigmoidLossrand3noise0p3.pth', './savemodels/LogisticLossrand3noise0p3.pth']
noises = [0, 0.3]
seeds = [7, 3]
widths = [5, 5]
depths = [6, 4]

losses = ['Logistic loss', 'Sigmoid loss']
getgrads = [getLogisticGrad, getSigmoidGrad]

fig2 = plt.figure(figsize = (18,6))
all_data = np.random.uniform(0, 1, size=(1000,1000))
hr = plt.contourf(all_data, cmap='jet', vmin=0, vmax=1)

for pthIndex in range(len(pthfiles)):
    fig = plt.figure(figsize = (100, 50))
    set_seed(seeds[pthIndex])
    Xm, Ym = getdatasets(noises[pthIndex])
    for lossIndex in range(len(losses)):

        output = getmodeloutput(pthfiles[pthIndex][lossIndex], widths[pthIndex], depths[pthIndex])

        Zp = getgrads[lossIndex](output, 1)
        Zn = getgrads[lossIndex](output, -1)
        
        plt.subplot(2, 3, lossIndex * 3 + 1)
        plt.contourf(X, Y, output, 1, cmap='ocean', vmin=-0.1, vmax=0.01)
        # cbar=plt.colorbar()
        plt.scatter(Xm[:,0], Xm[:,1], s=10, c=Ym)
        plt.title('Classification boundary of ' + losses[lossIndex].lower())

        plt.subplot(2, 3, lossIndex * 3 + 2)
        plt.contourf(X, Y, np.abs(Zp), cmap='jet', vmin=0, vmax=1)
        plt.title(losses[lossIndex] + ' with label 1')
        # plt.scatter(Xm[Ym==1,0], Xm[Ym==1,1], s=10, c='#fde725')

        plt.subplot(2, 3, lossIndex * 3 + 3)
        plt.contourf(X, Y, np.abs(Zn), cmap='jet', vmin=0, vmax=1)
        plt.title(losses[lossIndex] + ' with label -1')
        # plt.scatter(Xm[Ym==0,0], Xm[Ym==0,1], s=10, c='#440154')

        fig.subplots_adjust(right=0.9)

        #colorbar 左 下 宽 高 
        l = 0.92
        b = 0.12
        w = 0.015
        h = 1 - 2*b 

        #对应 l,b,w,h；设置colorbar位置；
        rect = [l,b,w,h] 
        cbar_ax = fig.add_axes(rect) 
        cb = plt.colorbar(hr, cax=cbar_ax)

        cb.set_label('Absolute value of gradient')

# plt.savefig('LosgisticAndSigmoidLossboundary.png', bbox_inches='tight', pad_inches=0.0)

plt.show()

'''
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 
 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 
 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 
 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 
 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 
 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 
 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 
 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 
 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 
 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 
 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 
 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 
'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
'''
