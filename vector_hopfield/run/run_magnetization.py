# %%
#######################################
home='/Users/matteo5/modern-hopfield/vector_hopfield'
import sys
sys.path.append(home)
#######################################
import src.hopfield as H
import numpy as np
#######################################

N_list=[500]
d_list=[2,3]

# np.random.seed(3)

## test stability of memories
n_samples = 10

for N in N_list:
    P_list=list(range(10,100,1))

    for d in d_list:
        for P in P_list:
            for sample_idx in range(n_samples):
                model = H.VectorHopfield(N,d,P)
                q, iter = model.run_dynamics(model.examples[0,:,:],verb=0)
                print("\t",N,d,P,q,iter,sample_idx,flush=True)
    


