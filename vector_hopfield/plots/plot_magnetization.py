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

#########################################################
data = np.loadtxt("../run/run_N500_d1.txt")

N_list=[500]
d_list=[1]
fmt_list = ["-"]

for n,N in enumerate(N_list):
    fmt = fmt_list[n]
    for d in d_list:
        # N,d,P,q,iter,sample_idx
        print(f"PLOTTING: {N},{d}")
        x, y_av, y_err, counts = average_cols(data, 2, 3, filter_cols=[0,1], filter_vals=[N,d])

        plt.errorbar(x/(N),y_av,y_err/np.sqrt(counts), fmt=fmt, capsize=2, linewidth=1, label=f"N={N} d={d}")
#########################################################
data = np.loadtxt("../run/run_N500_d2_3.txt")

N_list=[500]
d_list=[2,3]
fmt_list = ["-"]

for n,N in enumerate(N_list):
    fmt = fmt_list[n]
    for d in d_list:
        # N,d,P,q,iter,sample_idx
        print(f"PLOTTING: {N},{d}")
        x, y_av, y_err, counts = average_cols(data, 2, 3, filter_cols=[0,1], filter_vals=[N,d])

        plt.errorbar(x/(N),y_av,y_err/np.sqrt(counts), fmt=fmt, capsize=2, linewidth=1, label=f"N={N} d={d}")
#########################################################
data = np.loadtxt("../run/run_N5000_d4.txt")

N_list=[5000]
d_list=[4]
fmt_list = ["-"]

for n,N in enumerate(N_list):
    fmt = fmt_list[n]
    for d in d_list:
        # N,d,P,q,iter,sample_idx
        print(f"PLOTTING: {N},{d}")
        x, y_av, y_err, counts = average_cols(data, 2, 3, filter_cols=[0,1], filter_vals=[N,d])

        plt.errorbar(x/(N),y_av,y_err/np.sqrt(counts), fmt=fmt, capsize=2, linewidth=1, label=f"N={N} d={d}")
#########################################################
        
plt.vlines(0.141,  color="C0",ymin=0,ymax=1,linestyles="--",linewidth=1)
plt.vlines(0.037/2, color="C1",ymin=0,ymax=1,linestyles="--",linewidth=1)
plt.vlines(0.0113/2,color="C2",ymin=0,ymax=1,linestyles="--",linewidth=1)

plt.xscale("log")

plt.ylabel("magnetization")
plt.xlabel(r"$\alpha=\frac{P}{N}$")
plt.legend(fontsize=9)

plt.tight_layout()

plt.savefig("variousN.pdf")