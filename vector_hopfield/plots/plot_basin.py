#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cycler import cycler

############################################################################
# cm0=cm.tab10
# nc = 10
# cmax = 4
# plt.rc('axes', prop_cycle=(cycler('color', [cm0(c/(nc-1)) for c in range(cmax)])))
# # plt.rc('axes', prop_cycle=(cycler('color', [cm0(c/(nc-1)) for c in range(nc)])))

############################################################################

#########################################################
#########################################################
def average_cols(data, col_x, col_y, filter_cols=[], filter_vals=[]):

    print(data.shape)

    for k, filter_col in enumerate(filter_cols):
        filter_val = filter_vals[k]
        print(F"FILTERING: COL {filter_col} VAL {filter_val} ")
        mask = data[:,filter_col] == filter_val
        data = data[mask, :]
        print(data.shape)

    data_x = data[:,col_x]
    data_y = data[:,col_y]

    output_x, counts = np.unique(data_x, return_counts=True)
    print(counts)
    output_y = np.zeros(output_x.shape[0])
    error_y = np.zeros(output_x.shape[0])

    for i,x in enumerate(output_x):
        samples = data_y[data_x == x]
        output_y[i] = np.mean(samples)
        error_y[i] = np.std(samples)

    return output_x, output_y, error_y, counts

#########################################################
#########################################################
plt.figure(figsize=(4,3))
N=5000
P=40
#########################################################
for d in [1,2,3]:
    data = np.loadtxt(f"../run/basin_N{N}_d{d}_P{P}.txt")

    x, y_av, y_err, counts = average_cols(data, 3, 4, filter_cols=[0], filter_vals=[N])

    plt.errorbar(x,y_av,y_err/np.sqrt(counts), fmt="-", capsize=2, linewidth=1, label=f"P={P} d={d}")
#########################################################

# plt.xscale("log")

plt.ylabel("final mag")
plt.xlabel("initial mag")
plt.legend(title=f"N={N}",fontsize=9)

plt.tight_layout()

plt.savefig("basins.pdf")
# %%
