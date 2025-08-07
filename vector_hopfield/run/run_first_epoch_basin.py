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
P_list = [40]
d_list=[1,2,3,4]
q0_list=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
s_list = [1] ## order of the mixture

n_samples=1
seed = np.random.randint(2**32)
np.random.seed(seed)

which="tensor"
# which="vector"

home_dir="../"
results_dir="results/"
images_dir="images/"



for N in N_list:

    plt.clf()
    plt.figure(figsize=(4,3))

    for d in d_list:
        for P in P_list:
            for s in s_list:
                    for sample_idx in range(n_samples):
                        name="mag_1epoch_{}_N{:03d}_d{:d}_P{:03d}_s{:d}_{:02d}".format(which,N,d,P,s,sample_idx)
                        full_name = home_dir+results_dir+name+".txt"
                        outfile=open(full_name, "w")

                        print(f"\nwriting in {full_name}\n")
                        
                        q1_list = []
                        for q0 in q0_list:
                        
                            if which=="vector":
                                model = H.VectorHopfield(N,d,P)
                            elif which=="tensor":
                                model = H.TensorHopfield(N,d,P)
                            else:
                                raise NameError("uknown model type")

                            xi_list = [model.examples[i,:,:].copy() for i in range(s)]

                            coeff_list = [1]*s
                            ## build the mixtures ######################
                            if coeff_list==None:
                                coeff_list = [1]*len(xi_list)

                            mix = np.zeros(xi_list[0].shape)
                            for k,xi in enumerate(xi_list):
                                mix += xi * coeff_list[k]

                            mix /= np.linalg.norm(mix,axis=0,keepdims=True)
                            
                            ## add noise
                            noisy_mix = H.gen_rotated_conf(mix,q0)

                            qs_mix_0 = H.overlaps(noisy_mix,mix)
                            q_mix_0 = np.mean(qs_mix_0)

                            q, iter   = model.run_dynamics(
                                                            noisy_mix,
                                                            verb=0,     
                                                            max_iter=1,
                                                            stop_dq=1-1e-7,
                                                            syncronous=True,
                                                            )

                            dqs = H.overlaps(model.state,noisy_mix)
                            dq = np.mean(dqs)

                            qs_mix_1 = H.overlaps(model.state,mix)
                            q_mix_1 = np.mean(qs_mix_1)

                            q1_list.append(q_mix_1)

                            print(N, d, P, s, q_mix_0, q_mix_1,file=outfile)
                            print(N, d, P, s, q_mix_0, q_mix_1)

                        outfile.close()
                    
                        ## 
                        plt.plot(q0_list,q1_list,label=f"d={d}")

    name2="mag_1epoch_{}_N{:03d}_d{}_P{:03d}_s{:d}_{:02d}".format(which,N,"MANY",P,s,sample_idx)

    plt.title(name2,fontsize=8)
    plt.xlabel("initial overlap")
    plt.ylabel("final overlap")
    plt.legend()     

    plt.tight_layout()

    plt.savefig(home_dir+images_dir+name2+".pdf")

# print("\nseed:",seed)

# %%
