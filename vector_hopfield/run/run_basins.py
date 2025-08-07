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

N_list=[5000]
P_list = [10,20,40,80]

np.random.seed(3)

n_samples=1

d_list=[1,2,3]
q_list = np.arange(0.0, 1.1, 0.1)

for N in N_list:
    for d in d_list:
        for P in P_list:
            filename=f"basin_N{N}_d{d}_P{P}.txt"
            outfile=open(filename, "w")
            for sample_idx in range(n_samples):

                model = H.VectorHopfield(N,d,P)

                for q in q_list:
                    # q = round(q,2)
                    example = model.examples[0,:,:].copy()
                    model.randomize_state()
                    random_conf = model.state.copy()
                    noisy_example = H.gen_rotated_conf(example,q)

                    # noisy_example = q*example + (1-q)*random_conf
                    # noisy_example /= np.linalg.norm(noisy_example,axis=0,keepdims=True)

                    q_init = H.overlap(example,noisy_example)

                    q_noisy, iter = model.run_dynamics(noisy_example,syncronous=True,max_iter=1000)

                    q_fin = H.overlap(example,model.state)

                    print(N,d,P,q_init,q_fin, iter,flush=True)
                    print(N,d,P,q_init,q_fin, iter,flush=True,file=outfile)

            outfile.close()
    


