import glob
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import matplotlib.mlab as mlab


MIN_LAYERS_NUM = 0.1

input_folder = r"C:\Users\nnina\Desktop\Actin_stat"
filenames = glob.glob(input_folder + "/*.csv")
volume = []
print(filenames)
dfs = []
layers = []
cells = []
volumes = {1: 682, 10: 2487, 6: 1903, 7: 1576, 9:800}

norm_data = {}
layers_to_go = []
poisson_data = []

for img_path in filenames:
    df = pd.read_csv(img_path, usecols=["Number of fiber layers", "Actin Length", "Actin Volume"])
    dfs.append(df)
    parts = img_path.split('_')
    cell_no = int(parts[-1].split('.')[0])
    cells.append(cell_no)
    a = df["Actin Volume"].dropna().to_list()
    layers_to_go = [i for i in a if i > MIN_LAYERS_NUM]
    # layers_to_go = [i/volumes.get(cell_no) for i in a if i > MIN_LAYERS_NUM] # with correction to volume
    layers.append(layers_to_go)
    norm_data[cell_no] = norm.fit(layers_to_go) #(mu, sigma)
    # fit with curve_fit

n_bins = 20
colors = ['red', 'tan', 'lime', 'pink', 'black']
n, bins, patches = plt.hist(layers, n_bins, histtype='bar', color=colors, label=cells)
plt.legend(prop={'size': 10})

mu = 0
sigma = 0
# add a 'best fit' line
for i, cell_no in enumerate(cells):
    mu = norm_data.get(cell_no)[0]
    sigma = norm_data.get(cell_no)[1]
    y = norm.pdf( bins, mu, sigma)
    y = poisson.pmf(bins, 0.5)
    l = plt.plot(bins, y, 'r--', linewidth=2, color=colors[i])

#plot
# plt.xlabel('Layer num norm to nucleus volume')
# plt.xlabel('Volumes')
# plt.xlabel('Layer num')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.grid(True)
plt.show()