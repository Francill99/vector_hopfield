import os
import numpy as np
import sys

########################################################
## utility

def make_dir(path):
    if not os.path.exists(path): os.makedirs(path)
    
########################################################
## algebra for vector-spins
    
def overlap_matrix(S0, S1):
    return np.tensordot(np.transpose(S0),S1,axes=1)

def overlaps(S0, S1):
    return np.diagonal(overlap_matrix(S0, S1))

def overlap(S0, S1):
    return np.mean(np.diagonal(overlap_matrix(S0, S1)))    

def gen_rand_orth_conf(S):
    d,N = S.shape
    assert d>1

    rand_conf = np.random.randn(d,N)
    rand_conf /= np.linalg.norm(rand_conf,axis=0,keepdims=True)

    rand_orth_conf = np.zeros((d,N))

    ## TODO this for loop might be inefficient
    ## maybe use something like
    ## projector = np.tensordot(np.transpose(rand_conf),rand_conf,axes=0)
    for i in range(N):
        x = S[:,i].copy()
        parall_projector = np.tensordot(np.transpose(x),x,axes=0)

        r = rand_conf[:,i].copy()
        orthogonalizer = np.identity(d) - parall_projector

        orth_r = np.tensordot(r, orthogonalizer, axes=1)
        rand_orth_conf[:,i] = orth_r.copy()

    rand_orth_conf /= np.linalg.norm(rand_orth_conf,axis=0,keepdims=True)
    return rand_orth_conf

def gen_rotated_conf(S,q):
    d,N = S.shape

    if d>1:
        rand_orth_conf = gen_rand_orth_conf(S)
        noisy_S = np.sqrt(1-q*q) * rand_orth_conf + q * S

    else:
        noisy_S = S.copy()
        dist = (1-q)/2
        n_to_flip = int(dist*N)
        mask = np.random.choice(N,n_to_flip,replace=False)
        noisy_S[:,mask] *= -1
        
    return noisy_S

########################################################
## classes

class VectorHopfield:
    def __init__(self, N, d, P):
        self.N = N # number of tokens (spins)
        self.d = d # dimension of spins
        self.P = P # number of examples
        self.state = np.zeros((self.d,self.N)) # state matrix
        self.randomize_state()

        self.examples = np.zeros((self.P,self.d,self.N)) # examples tensor
        self.gen_examples()

        self.J = np.zeros((self.N,self.N)) # coupling tensor
        self.gen_J() 

    def gen_examples(self):
        examples = np.random.randn(self.P,self.d,self.N)
        Z = np.linalg.norm(examples,axis=1,keepdims=True)
        examples /= Z
        self.examples = examples
        return

    def gen_J(self):
        # tensor = np.zeros((self.d,self.N,self.d,self.N))
        for mu in range(self.P):
            print(f"{mu}/{self.P}",end="\r",flush=True,file=sys.stderr)
            example = self.examples[mu,:,:]
            self.J += np.tensordot(np.transpose(example),example,axes=1)
        
        print("\n",flush=True,file=sys.stderr)
        # tensor 
        self.J /= self.N

        for i in range(self.N):
            self.J[i,i] = 0

        return 

    def normalize_spins(self):
        self.state /= np.linalg.norm(self.state,axis=0,keepdims=True)
        return

    def get_random_state(self):
        state = np.random.randn(self.d,self.N)
        return state / np.linalg.norm(state,axis=0,keepdims=True)

    def randomize_state(self):
        self.state = np.random.randn(self.d,self.N)
        self.normalize_spins()
        return self.state.copy()
    
    ## syncronous version ######################
    def compute_local_fields(self):
        return np.tensordot(self.state, self.J, axes=1)

    def update_spins(self, ORR=0):
        initial_state = self.state.copy()
        h = self.compute_local_fields()
        ## XXX is this correct? does it destabilize mixtures?
        h += 1e-12*np.random.randn(*h.shape)
        h_hat = h/np.linalg.norm(h,axis=0,keepdims=True)
        ## OVER-RELAXATION in the vector case
        if self.d>1:
<<<<<<< Updated upstream
            sr=np.dot(h/np.linalg.norm(h,axis=0,keepdims=True), 2*np.diag(np.diag(np.tensordot(np.transpose(self.state), h/np.linalg.norm(h,axis=0,keepdims=True), axes=1))))-self.state
=======
            sr=np.dot(h_hat, 
                      2*np.diag(np.diag(np.tensordot(np.transpose(self.state), h_hat, axes=1))))-self.state
>>>>>>> Stashed changes
            h += ORR*sr
        new_state = h / np.linalg.norm(h,axis=0,keepdims=True)
        self.state = new_state
        return overlap(self.state,initial_state)
      
    ############################################

    ## Asyncronous version ######################
    def compute_local_field(self,i):
        return np.tensordot(self.state, self.J[:,i], axes=1)
    
    def update_spin(self, i, ORR=0):
        initial_spins = self.state.copy()
        h = self.compute_local_field(i)
        h += 1e-12*np.random.randn(*h.shape)
        ## OVER-RELAXATION in the vector case
        if self.d>1:
            sr=2*np.dot(self.state[:,i], h/np.linalg.norm(h,axis=0,keepdims=True))*h/np.linalg.norm(h,axis=0,keepdims=True)-self.state[:,i]
            h += ORR*sr
        new_spin = h / np.linalg.norm(h,axis=0,keepdims=True)
        self.state[:,i] = new_spin
        return overlap(self.state,initial_spins)

    ############################################
    def run_dynamics(self, initial_state, max_iter = 2000, stop_dq = 0.99999999,verb=0, ORR=0, syncronous=False):
        self.state = initial_state.copy()
        q = None

        for iter in range(max_iter):

            if syncronous:
                dq = self.update_spins(ORR)
            else:
                old_state = self.state.copy()
                for i in np.random.permutation(self.N):
                    _ = self.update_spin(i, ORR)

                dq = overlap(self.state, old_state)

            if verb>0:
                filename=f"relax_dynamics/dyn_N{self.N}_d{self.d}_P{self.P}.txt"
                q = overlap(self.state, initial_state)
                print(iter, q, dq, file=open(filename, "a"), flush=True)

            if(dq>stop_dq):
                q = overlap(self.state, initial_state)
                break

        return q, iter   

    def get_mix(self,idx_list):

        mix = np.zeros(self.examples[0,:,:].shape)
        for idx in idx_list:
            mix += self.examples[idx,:,:]
        mix /= np.linalg.norm(mix,axis=0,keepdims=True)

        return mix

    ############################################
    ## we have a separate method for mixtures 
    ## because we keep track of mag. for the components of the mix too
    ## therefore we need the list of components
    def run_dynamics_mixture(self, xi_list, coeff_list=None,
                             max_iter = 200, 
                             stop_dq = 0.9999,
                             verb=0, ORR=0, 
                             syncronous=True,
                             outfile=None,
                             q0=1
                             ):
        
        mix_order = len(xi_list)

        ## build the mixtures ######################
        if coeff_list==None:
            coeff_list = [1]*len(xi_list)

        mix = np.zeros(xi_list[0].shape)
        for k,xi in enumerate(xi_list):
            mix += xi * coeff_list[k]

        mix /= np.linalg.norm(mix,axis=0,keepdims=True)
        
        ## add noise
        noisy_mix = gen_rotated_conf(mix,q0)
        self.state = noisy_mix

        q_mix = overlap(self.state, mix)

        ## set up list of overlaps
        q_list = []
        for s in range(mix_order):
            q_list.append(overlap(self.state,xi_list[s])) 

        # ## TEST ONE SUBMIX
        # submix = np.zeros(xi_list[0].shape)
        # for k in range(3):
        #     submix += xi_list[k]

        # submix /= np.linalg.norm(submix,axis=0,keepdims=True)
        # q_list.append(overlap(self.state,submix)) 
        # ######################

        iter = 0
        dq = 0
        ## print before dynamics
        if verb>0:
            print(iter, dq, q_mix, *q_list)
            if outfile!=None:
                print(iter, dq, q_mix, *q_list ,flush=True,file=outfile)

        ## dynamics
        for iter in range(1,max_iter+1):

            ## actual update
            dq = self.update_spins(ORR)

            ## compute overlaps
            q_mix = overlap(self.state, mix)
            for s in range(mix_order):
                q_list[s] = overlap(self.state,xi_list[s])

            # q_list[s+1] = overlap(self.state,submix)

            ## print during dynamics
            if verb>0:
                print(iter, dq, q_mix, *q_list)
                if outfile!=None:
                    print(iter, dq, q_mix, *q_list,flush=True,file=outfile)

            ## stopping criterion
            if(dq>stop_dq):
                q_mix = overlap(self.state, mix)
                break

        return iter, dq, q_mix, *q_list  

    ############################################

    def run_dynamics_mixture_async(self, xi_list, coeff_list=None,
                             max_iter = 200, 
                             stop_dq = 0.9999,
                             verb=0, ORR=0, 
                             syncronous=True,
                             outfile=None,
                             q0=1
                             ):
        
        mix_order = len(xi_list)

        ## build the mixtures ######################
        if coeff_list==None:
            coeff_list = [1]*len(xi_list)

        mix = np.zeros(xi_list[0].shape)
        for k,xi in enumerate(xi_list):
            mix += xi * coeff_list[k]

        mix /= np.linalg.norm(mix,axis=0,keepdims=True)
        
        ## add noise
        noisy_mix = gen_rotated_conf(mix,q0)
        self.state = noisy_mix

        q_mix = overlap(self.state, mix)

        ## set up list of overlaps
        q_list = []
        for s in range(mix_order):
            q_list.append(overlap(self.state,xi_list[s])) 

        ## TEST ONE SUBMIX
        # submix = np.zeros(xi_list[0].shape)
        # for k in range(3):
        #     submix += xi_list[k]

        # submix /= np.linalg.norm(submix,axis=0,keepdims=True)
        # q_list.append(overlap(self.state,submix)) 
        ######################

        # iter = 0
        # dq = 0
        # ## print before dynamics
        # if verb>0:
        #     print(iter, dq, q_mix, *q_list)
        #     if outfile!=None:
        #         print(iter, dq, q_mix, *q_list ,flush=True,file=outfile)

        ## dynamics
        for iter in range(0,max_iter):

            ## actual update
            # dq = self.update_spins(ORR)
            old_state = self.state.copy()
            ## local updates
            local_iter = 0
            for i in np.random.permutation(self.N):
                dq = self.update_spin(i, ORR)
                local_iter += 1
                ## compute overlaps
                q_mix = overlap(self.state, mix)
                for s in range(mix_order):
                    q_list[s] = overlap(self.state,xi_list[s])

                # q_list[s+1] = overlap(self.state,submix)

                ## print during dynamics
                if verb>0:
                    time = iter + float(local_iter)/self.N
                    print(time, dq, q_mix, *q_list)
                    if outfile!=None:
                        print(time, dq, q_mix, *q_list,flush=True,file=outfile)

            dq = overlap(self.state, old_state)
            ## stopping criterion
            if(dq>stop_dq):
                q_mix = overlap(self.state, mix)
                break

        return iter, dq, q_mix, *q_list  

    ############################################
    ## 1) dividing the NLpL by P makes it so that the energy is O(N) and not O(P*N)
    ## 2) for convenience, we print E/N. 
    ## 3) the NLpL of the Hebb rule is -1 (approximately if finite size)
    ## 4) normalizing the local fields makes it so that NLpL=1 even at finite size

    def compute_energy(self):
        h = self.compute_local_fields()
        h += 1e-12*np.random.randn(*h.shape)
        #h /= np.linalg.norm(h,axis=0,keepdims=True)
        return -np.tensordot(self.state,h,axes=2)/2
    
    ## right now, compute_NLpL is NOT the actual NLpL
    ## since it's missing the reaction term
    ## i.e. -log(Zi)
    def compute_reaction_term(self):
        return

    def compute_NLpL(self):
        pseudo_likelihood = 0
        for mu in range(self.P):
            self.state = self.examples[mu,:,:]
            pseudo_likelihood += self.compute_energy() / self.P

        return pseudo_likelihood
    
    def print_local_fields(self, filename, ov, iter, sample_index, strOv):
        
        local_fields = self.compute_local_fields()
        
        stream = open(filename, "a")
        
        print(ov, iter, self.compute_energy(), sample_index, 0, strOv, file=stream)
           
        for i in range(self.N):
            st=""
            for j in range(self.d):
                st=st+str(local_fields[j, i])+" "
            
            print(st, sample_index, 1, file=stream)
                
        stream.close()

        return
    
########################################################
class TensorHopfield(VectorHopfield):
    def __init__(self, N, d, P):
        super().__init__(N, d, P)

    def gen_J(self):
        self.J = np.zeros((self.d,self.N,self.d,self.N))
        for mu in range(self.P):
            print(f"{mu}/{self.P}",end="\r",flush=True,file=sys.stderr)
            example = self.examples[mu,:,:]
            self.J += np.tensordot(example,example,axes=0)
        self.J /= self.N
        #self.J /= (self.N*self.d) parliamone

        ##zero diag
        for i in range(self.N):
             self.J[:,i,:,i] = 0

        return 

    ## Asyncronous version ######################
    ## NOTE this is inefficient?
    def compute_local_field(self,i):
        return np.tensordot(self.J[:,i,:,:], self.state, axes=2)

    ## syncronous version ######################
    def compute_local_fields(self):
        return np.tensordot(self.J, self.state, axes=2) ## note the 2
    ############################################

########################################################

# Hessian spectrum of the Hopfield model

class spectrumHopfield:
    
    def __init__(self, model):
        
        self.conf=model.state ## the configuration on which the Hessian is evaluated
        self.couplings=model.J ## the coupling matrix
         
        self.localFieldsMagn=np.linalg.norm(model.compute_local_fields(), axis=0, keepdims=False) ## the norms of the local fields
                
        self.examples=model.examples
        
        self.d=np.shape(self.conf)[0]
        self.N=np.shape(self.conf)[1]
        self.P=np.shape(self.examples)[0]
        
        self.random_ortho_bases=self.generate_random_orthogonal_bases()
        
        hss=np.zeros((self.d-1, self.N, self.d-1, self.N))
        ''' aux=np.zeros((self.d-1, self.N, self.d-1, self.N)) '''
        JJ=np.zeros((self.d, self.N, self.d, self.N))
        
        if np.shape(self.couplings) == (self.N, self.N): ## isotropic couplings
                        
            for i in range(self.d):
            
                JJ[i,:,i,:] = self.couplings
         
         
        else: ## any other form of interactions
            
            JJ = self.couplings
                    
        for a in range(self.d-1): ## partially vectorized in the sites but not memory demanding (gets O(N^2) memory in intermediate passages)
            for b in range(self.d-1):
                
                for site in range(self.N):
                    
                    aux = np.dot(JJ[:,:,:,site], self.random_ortho_bases[b, :, site])
                        
                    hss[a, :, b, site] = -np.diag(np.dot(np.transpose(self.random_ortho_bases[a,:,:]), aux))+np.diag(self.localFieldsMagn)[:, site]*(a==b)

        
        tmp = np.zeros(((self.d-1)*self.N, (self.d-1)*self.N))
        
        for a in range(self.d-1):
            for b in range(self.d-1):
                for i in range(self.N):
                    for j in range(self.N):
                        
                        tmp[i*(self.d-1)+a, j*(self.d-1)+b] = hss[a, i, b, j]
                        
        self.Hessian = tmp
           
        s = self.diagonalize()
        
        self.eigenvalues = s.eigenvalues
        self.eigenvectors = s.eigenvectors
       
    def generate_random_orthogonal_bases(self): ## generate random bases in each of the d-1--dimensional subspaces orthogonal to the S_i's
        
        us=np.random.randn(self.d-1, self.d, self.N)
        us /= np.linalg.norm(us, axis=1, keepdims=True)
        
        ## Gram-Schmidt
        
        oldus = np.copy(us)
        
        for i in range(self.N): ## non sono riuscito a farlo funzionare vettorizzato sui siti
        
            for a in range(self.d-1):
            
                us[a,:,i] -= np.dot(oldus[a,:,i], self.conf[:, i])*self.conf[:,i]

                for b in range(a):
                    
                    us[a,:,i] -= np.dot(oldus[a,:,i], us[b,:,i])*us[b,:,i]
                
                us[a,:,i] /= np.linalg.norm(us[a,:,i], axis=0, keepdims=True)
        
        ''' for i in range(self.d-1): versione vettorizzata sui siti che non funziona
            
            us[i,:,:] -= np.matmul(self.conf, np.diag(np.diag(np.tensordot(np.transpose(self.conf), oldus[i,:,:], axes=1))))
            
            for j in range(i):
                
                us[i,:,:] -= np.matmul(us[j,:,:], np.diag(np.diag(np.tensordot(np.transpose(us[j,:,:]), oldus[i,:,:], axes=1))))
                
            us[i,:,:] /= np.linalg.norm(us[i,:,:], axis=1, keepdims=True) '''
         
        
        return us

    
    def diagonalize(self):
        
        return np.linalg.eigh(self.Hessian)
        
    def compute_eigenvalues_spacings(self):
        
        return self.eigenvalues[1:]-self.eigenvalues[:(self.d-1)*self.N]
    
    def compute_r(self): ## r is a parameter that provides information on the statistics of the spacing: average r equal to 0.383 indicates a Poissonian point statistics, 0.529 a Wigner-Dyson one
        
        sp=self.eigenvalues[1:]-self.eigenvalues[:(self.d-1)*self.N]
        
        tmp=np.reshape(self.eigenvalues, (self.d-1, self.N))
        
        s=np.mean(tmp, axis=0, keepdims=False)
        
        return [np.maximum(sp[:(self.d-1)*self.N-1], sp[1:])/np.minimum(sp[:(self.d-1)*self.N-1], sp[1:]), np.maximum(s[:self.N-1], s[1:])/np.minimum(s[:self.N-1], s[1:])]
    
    def compute_eigenvectors_weights(self):
        
        eigvecs_reshaped = np.reshape(self.eigenvectors, (self.N, self.d-1, (self.d-1)*self.N))
        
        return np.linalg.norm(eigvecs_reshaped, axis=1, keepdims=False)
        
    
    def compute_invariant_eigenvector_moments(self, k=2): ##computes moments from the third to the (3+k)th (the second moment is the normalization, the fourth the IPR)
        
        iprs = np.zeros((k, self.N*(self.d-1)))
        
        eigvecs_reshaped = np.reshape(self.eigenvectors, (self.N, self.d-1, (self.d-1)*self.N))
        
        aux = np.linalg.norm(eigvecs_reshaped, axis=1, keepdims=False)
        
        for i in range(k):
            
            iprs[i] = np.sum(aux**(3+i), axis=0, keepdims=False)
        
        return iprs
    
    ''' def compute_invariant_eigenvector_moments_babbeo_check(self, k=2): ##check con codice inefficiente per verificare che le vettorizzazioni nel codice efficiente siano fatte bene
        
        iprs = np.zeros((k, self.N*(self.d-1)))
        
        aux = np.zeros((self.N*(self.d-1), self.N))
        
        for ii in range((self.d-1)*self.N):
            for i in range(self.N):
                for a in range(self.d-1):
                    
                    aux[ii, i] += self.eigenvectors[i*(self.d-1)+a, ii]**2
                  
                aux[ii, i] = np.sqrt(aux[ii, i])
                
        
        for i in range(k):
            
            iprs[i] = np.sum(aux**(3+i), axis=1, keepdims=False)
        
        return iprs '''
    
    def change_eigenvector_representation(self): ## represents eigenvectors in the canonical base of R^{N\times d}
        
        v = np.zeros((self.d, self.N, (self.d-1)*self.N))
        
        tmp = np.reshape(self.eigenvectors, (self.N, self.d-1, (self.d-1)*self.N))
        
        for i in range(self.N*(self.d-1)):
            
            v[:, :, i] = np.diagonal(np.tensordot(self.random_ortho_bases, tmp[:,:,i], axes=([0], [1])), axis1=1, axis2=2)
            
        return v
    
    def compute_overlap_eigenvectors_examples(self): ## overlaps of eigenvectors with the examples
        
        a=self.change_eigenvector_representation()
        
        return np.tensordot(a, self.examples, axes=([0, 1], [1, 2]))/np.sqrt(self.N)
    
    def print_eigenvalues_and_iprs(self, iprs, filename, k, sample_index):
        
        stream=open(filename, "a")
        
        for i in range(self.N*(self.d-1)):
            
            st=""
            
            for j in range(k):
                
                st = st+str(iprs[j, i])+" "
        
            print("%d %lf %s%d"%(i, self.eigenvalues[i], st, sample_index), file=stream)
            
        stream.close()
        
        return
    
    def print_overlaps_eigenvectors_examples(self, ovs, filename, sample_index, r, e):
        
        stream=open(filename, "a")
        
        for i in range(self.P):
                
            print(i, r[i], e[i], ovs[i], sample_index, file=stream)
                
        stream.close()
        
        return
    
    def get_where_eigenvector_weight_is_maximum(self, eig_rank):
        
        eigvec_reshaped = np.reshape(self.eigenvectors[:,eig_rank], (self.N, self.d-1))
        
        aux = np.linalg.norm(eigvec_reshaped, axis=1, keepdims=False)
    
        mx = np.max(aux)
        
        return [mx**2, np.where(aux==mx)[0][0]]
        
        