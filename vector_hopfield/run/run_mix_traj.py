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

N_list=[2000]
P_list = [20,80]
d_list=[3]
q0_list=[0.3]
# s_list = [2,3,4,5,6,7] ## order of the mixture
s_list = [5] ## order of the mixture

n_samples=10
seed = np.random.randint(2**32)
np.random.seed(seed)

which="tensor"
# which="vector"

home_dir="../"
results_dir="results/"

for N in N_list:
    for d in d_list:
        for P in P_list:
            for s in s_list:
                for q0 in q0_list:
                    for sample_idx in range(n_samples):

                        if which=="vector":
                            model = H.VectorHopfield(N,d,P)
                        elif which=="tensor":
                            model = H.TensorHopfield(N,d,P)
                        else:
                            raise NameError("uknown model type")

                        xi_list = [model.examples[i,:,:].copy() for i in range(s)]

                        filename="mag_mix_{}_N{:03d}_d{:d}_P{:03d}_s{:d}_{:02d}.txt".format(which,N,d,P,s,sample_idx)
                        outfile=open(home_dir+results_dir+filename, "w")

                        coeff_list = [1]*s
                        # coeff_list = [1,0,0,0,0]
                        # coeff_list = [1.2,1.2,1.2,1,1]

                        output = model.run_dynamics_mixture(
                                                        xi_list,
                                                        coeff_list=coeff_list,
                                                        verb=1,     
                                                        max_iter=1000,
                                                        stop_dq=1-1e-7,
                                                        # syncronous=False, ## TODO
                                                        outfile=outfile,
                                                        q0=q0
                                                        )

                        ###########
                        mix = np.zeros(xi_list[0].shape)
                        for k,xi in enumerate(xi_list):
                            mix += xi * coeff_list[k]
                        mix /= np.linalg.norm(mix,axis=0,keepdims=True)

                        ###########
                        # submix = np.zeros(xi_list[0].shape)
                        # for k in range(3):
                        #     submix += xi_list[k]

                        # submix /= np.linalg.norm(submix,axis=0,keepdims=True)

                        # print()
                        # print(H.overlap(mix,mix))
                        # print(H.overlap(submix,submix))
                        # print(H.overlap(mix,submix))
                        # print()
                        # print(H.overlap(xi_list[0],mix))
                        # print(H.overlap(xi_list[1],mix))
                        # print(H.overlap(xi_list[2],mix))
                        # print(H.overlap(xi_list[3],mix))
                        # print(H.overlap(xi_list[4],mix))
                        # print()
                        # print(H.overlap(xi_list[0],submix))
                        # print(H.overlap(xi_list[1],submix))
                        # print(H.overlap(xi_list[2],submix))
                        # print(H.overlap(xi_list[3],submix))
                        # print(H.overlap(xi_list[4],submix))            
                        # print()

                        q_list = []
                        for i in range(P):
                            overlap = H.overlap(model.state, model.examples[i,:,:])
                            print(i,overlap)
                            q_list.append(overlap)

                        q_list = np.array(q_list)

                        print("\n",np.sort(q_list[::-1]))

                        outfile.close()
        

print("\n",seed)



# %%
