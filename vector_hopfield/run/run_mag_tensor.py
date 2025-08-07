# %%
#######################################
home='/Users/matteo5/modern-hopfield/vector_hopfield'
import sys
sys.path.append(home)
#######################################
import src.hopfield as H
import numpy as np
import matplotlib.pyplot as plt
#######################################

N_list=[1000]
P_list = list(range(10,300,10))

np.random.seed(3)

n_samples=10

d_list=[2,3]

for N in N_list:
    for d in d_list:
        filename=f"mag_tensor_N{N}_d{d}.txt"
        outfile=open(filename, "w")
        for P in P_list:
            for sample_idx in range(n_samples):

                # model = H.VectorHopfield(N,d,P)
                # print(model.J.shape)
                # print(model.compute_local_fields().shape)

                model = H.TensorHopfield(N,d,P)
                # print(model.J.shape)
                # print(model.compute_local_fields().shape)

                example = model.examples[0,:,:].copy()
                m, iter = model.run_dynamics(example,syncronous=True,max_iter=1000)
                print(N,d,P,m,iter,flush=True)
                print(N,d,P,m,iter,flush=True,file=outfile)

        outfile.close()
        

    


