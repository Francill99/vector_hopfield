import numpy as np
import src.hopfield as H

Nlist=[50, 100, 200]
alpha=0.1
d=3

nsample=10
dorr=0.25
ndorr=10

for N in Nlist:
    P=int(alpha*N)
    path=f"orr_test_N{N}_d{d}_P{P}.txt"
    for n in range(nsample):
        
        orr=0.
        
        model=H.VectorHopfield(N, d, P)
        
        ini=model.get_random_state()
        
        for i in range(ndorr):
        
            q, iter = model.run_dynamics(ini, max_iter=1000, ORR=orr)
            
            print(orr, iter, model.compute_energy()/N, n, file=open(path, "a"), flush=True)
            
            orr += dorr
        
    
    