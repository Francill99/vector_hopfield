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
d_list=[2]

np.random.seed(3)

P_list = [20]

# print("#l E_init/N E_fin/N q q_init_1 q_init_2 q_fin_1 q_fin_2 iter")

for N in N_list:
    for d in d_list:
        for P in P_list:
            for sample_idx in range(1):

                model = H.VectorHopfield(N,d,P)

                example_1 = model.examples[0,:,:]
                example_2 = model.examples[1,:,:]
                # model.randomize_state()
                # example_2 = model.state.copy()

                for l in np.arange(0.0, 1.1, 0.1):

                    l = round(l,4)

                    probe = (1-l)*example_1 + l*example_2
                    probe /= np.linalg.norm(probe,axis=0,keepdims=True)

                    model.state = probe
                    E_init = model.compute_energy()

                    q, iter = model.run_dynamics(probe,syncronous=True)

                    q1_init = H.overlap(probe,example_1)
                    q2_init = H.overlap(probe,example_2)

                    q1_fin = H.overlap(model.state,example_1)
                    q2_fin = H.overlap(model.state,example_2)

                    E_fin = model.compute_energy()

                    # print(N,d,P,l,q,iter,sample_idx,flush=True)
                    # print(l, q,q_init_1,q_init_2,q_fin_1,q_fin_2, iter,flush=True)
                    print(l, q, q1_init, q1_fin, q2_init, q2_fin, E_init/N, E_fin/N, iter,flush=True)
                    # print( np.round(l,2)  ,E_init/N,E_fin/N,q,q_fin_1,q_fin_2, iter,flush=True)
    


