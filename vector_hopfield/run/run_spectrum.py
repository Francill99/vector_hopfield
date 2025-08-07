#home='/Users/fnicoletti/DottoratoFisica/Hopfield'

import src.hopfield as H
import sys
#sys.path.append(home)
import numpy as np

## This python script computes and analyses the full spectrum of multiple vector hopfield samples, where the default model to be studied is the Tensor Hopfield.
# 
# The functions used in this code are contained in the hopfield class, in the file "src/hopfield.py".

# It first runs a syncronous minimisation dynamics, based on moves of spins alignments to their local fields (run.dynamics(...) method).
# It diagonalises the Hessian matrix evaluated on the stationary point reached by run.dynamics(...), through the instantiation of a spectrumHopfield(...) object.
# It analyses the spectrum
# 1) Computes (Inverse Participation Ratios) IPRs of several degrees, to quantify concentration of eigenvector weights. Uses compute_invariant_eigenvector_moments(...) method.
# 1.5) Generates scatter plots of eigenvector weights and local fields, local magnetisation and eigenvalues, to be used in the paper.
# 2) Computes the overlap between patterns and low-rank eigenvectors. Uses compute_overlap_eigenvector_examples(...) method. NOT ANYMORE

## primary input parameters

## N: system size, INTEGER
## d: spin dimensions, INTEGER
## alpha: ratio between number of patterns and system size, \alpha = \frac{P}{N}, FLOAT
## option_initialization: specify the category of the initial condition, STRING
## "R" is random, "C" is for an initial state correlated to a pattern, "M" for one correlated with a mixture
## sample_index_start: the starting sample index for the batch, equal to b*nsample, being b=0,...,nb-1 the batch index, INTEGER
## This last input value is automatically filled by the batch process that launches the .py script in parallel
## nsample: number of samples per batch, INTEGER

## secondary input (directly derived from primary)

## P: the number of patterns

## output: for each sample

# Prints the components of the local fields \vec{eta}_i.
# Prints eigenvalues and related IPRS, in ascending rank order.
# Prints, for each pattern, the maximal overlap with eigenvectors, together with the rank of the mode with such an overlap. NOT ANYMORE (COMMENTED OUT)

file_input="input_for_run_spectrum.txt"

stream=open(file_input, "r")

input=stream.readline().split(" ")

N=int(input[0]);
d=int(input[1]);
alpha=float(input[2]);
option_initialization=input[3]
sample_index_start=int(input[4]);
nsample=int(input[5]);

P=int(alpha*N+0.5);

stream.close()

if option_initialization == "R": ## random initial condition (spin glass states)

    for i in range(sample_index_start, sample_index_start+nsample):
        
        ## instantiation and initialisation
        
        model = H.TensorHopfield(N, d, P)
        
        ini = model.get_random_state()
        
        filename_local_fields = f"final_states/final_states_N{N}_d{d}_P{P}_farFromPatterns.txt"
        filename_spectrum = f"spectra/spectrum_and_iprs_N{N}_d{d}_P{P}_farFromPatterns.txt" ## comment for earlier version of Python
        
        ''' filename_local_fields = "final_states/final_states_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_farFromPatterns.txt"
        filename_spectrum = "spectra/spectrum_and_iprs_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_farFromPatterns.txt" ''' ##uncomment for earlier version of Python
        
        ## minimisation dynamics
                
        q, iter=model.run_dynamics(ini, ORR=0.0, syncronous=True, max_iter=10000, stop_dq=1.-10**(-12))
        
        model.print_local_fields(filename_local_fields, q, iter, i)
        
        ## diagonalisation and spectral analysis
        
        spectrum = H.spectrumHopfield(model)
        
        iprs=spectrum.compute_invariant_eigenvector_moments(4)
            
        spectrum.print_eigenvalues_and_iprs(iprs, filename_spectrum, 4, i)
        
        '''overlaps_eigenvectors_examples = spectrum.compute_overlap_eigenvectors_examples()
        
        filename_overlaps = f"spectra/overlap_eigenvector_example_N{N}_d{d}_P{P}_farFromPatterns.txt"
        
        ## filename_overlaps = "spectra/overlap_eigenvector_example_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_farFromPatterns.txt" uncomment for earlier version of Python
        
        ranks_of_maximum_overlap_per_example = np.argmax(overlaps_eigenvectors_examples, axis=0)
        
        eigs_of_maximum_overlap_per_example = spectrum.eigenvalues[ranks_of_maximum_overlap_per_example]
        
        spectrum.print_overlaps_eigenvectors_examples(np.max(
        overlaps_eigenvectors_examples, axis=0), filename_overlaps, i,
          ranks_of_maximum_overlap_per_example, eigs_of_maximum_overlap_per_example)
            ## for any pattern, prints the maximal overlap '''

elif option_initialization == "C": ## pattern-correlated initial condition (pattern states)
    
    for i in range(sample_index_start, sample_index_start+nsample):
        
        ## instantiation and initialisation
        
        model = H.TensorHopfield(N, d, P)
        
        ini = model.examples[0,:,:]
        
        filename_local_fields = f"final_states/final_states_N{N}_d{d}_P{P}_CloseToPattern.txt"
        filename_spectrum = f"spectra/spectrum_and_iprs_N{N}_d{d}_P{P}_CloseToPattern.txt"
        
        ''' filename_local_fields = "final_states/final_states_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseToPattern.txt"
        filename_spectrum = "spectra/spectrum_and_iprs_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseToPattern.txt" '''
        
        ## minimisation dynamics
            
        q, iter=model.run_dynamics(ini, ORR=0.0, syncronous=True, max_iter=10000, stop_dq=1.-10**(-12))
        
        model.print_local_fields(filename_local_fields, q, iter, i)
        
        ## diagonalisation and spectral analysis
        
        spectrum = H.spectrumHopfield(model)
        
        iprs=spectrum.compute_invariant_eigenvector_moments(4)
        
        spectrum.print_eigenvalues_and_iprs(iprs, filename_spectrum, 4, i)
        
        '''overlaps_eigenvectors_examples = spectrum.compute_overlap_eigenvectors_examples()
        
        filename_overlaps = f"spectra/overlap_eigenvector_example_N{N}_d{d}_P{P}_CloseToPattern.txt"
        
        ## filename_overlaps = "spectra/overlap_eigenvector_example_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseToPattern.txt"
        
        ranks_of_maximum_overlap_per_example = np.argmax(overlaps_eigenvectors_examples, axis=0)
        
        eigs_of_maximum_overlap_per_example = spectrum.eigenvalues[ranks_of_maximum_overlap_per_example]
        
        spectrum.print_overlaps_eigenvectors_examples(np.max(overlaps_eigenvectors_examples,
          axis=0), filename_overlaps, i, ranks_of_maximum_overlap_per_example,
            eigs_of_maximum_overlap_per_example) 
            ## for any pattern, prints the maximal overlap'''
         
        ''' decomment above to study overlap of patterns with low rank eigenvectors '''
        
        ## below: print data points for local fields, eigenvectors weights and local magnetisation
        ## and eigenvalues. Needed for scatter plots in the paper

        lf = np.linalg.norm(model.compute_local_fields(), axis=0, keepdims=False)
        
        qq = model.get_local_magn()
        
        ## rank = 0
        
        rank = np.int64(P/2)
        
        ww = spectrum.compute_eigenvectors_weights(int(2*rank))
        
        fname = "spectra/local_noise_and_eigv_weights_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseToPattern_nontrivial_bulk.txt"
        
        eig = spectrum.eigenvalues[rank]
        
        for i in range(N):
            
            print(i, lf[i], ww[i,rank], qq[i], eig, q, file=open(fname, "a"))
        
        

elif option_initialization == "M": ## mixture-correlated initial condition (mixture states)
    
    s = 3 ## order of mixture (must be less than equal to P, the number of patterns)
    aS = 1. ## asymmetry parameter (Equal to 1 if mixture is symmetric, to be kept strictly non-zero)
    
    if s>P:
        
        sys.exit("Order of mixture larger than number of patterns.")
    
    if aS==0:
        
        sys.exit("Asymmetry parameter must be non-zero.")
    
    if aS != 1:
        
        typeOfMixture = "aSym"+str(s)
        
    else:
        
        typeOfMixture = "Sym"+str(s)
    
    for i in range(sample_index_start, sample_index_start+nsample):
        
        ## instantiation and initialisation
        
        model = H.TensorHopfield(N, d, P)
        
        X = np.zeros((s, d, N))
        
        for pattern in range(s-1):
            
            X[pattern] = model.examples[pattern,:,:]
        
        X[s-1] = aS*model.examples[s-1,:,:]
        
        mix = np.sum(X, axis=0)
        ini = mix/np.linalg.norm(mix, axis=0, keepdims=True)
        
        filename_local_fields = f"final_states/final_states_N{N}_d{d}_P{P}_CloseTo{typeOfMixture}Mixture.txt"
        filename_spectrum = f"spectra/spectrum_and_iprs_N{N}_d{d}_P{P}_CloseTo{typeOfMixture}Mixture.txt"
        
        ''' filename_local_fields = "final_states/final_states_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseTo"+str(typeOfMixture)+"Mixture.txt"
        filename_spectrum = "spectra/spectrum_and_iprs_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseTo"+str(typeOfMixture)+"Mixture.txt" '''
        
        ## minimisation dynamics
            
        q, iter = model.run_dynamics(ini, ORR=0.0, syncronous=True, max_iter=10000, stop_dq=1.-10**(-2))
        
        strOv = ""
        
        for pattern in range(s):
            
            strOv = strOv+str(H.overlap(model.state, X[pattern]))+" "
            
        model.print_local_fields(filename_local_fields, q, iter, i, strOv)
        
        ## diagonalisation and spectral analysis
        
        spectrum = H.spectrumHopfield(model)
        
        iprs=spectrum.compute_invariant_eigenvector_moments(4)
        
        spectrum.print_eigenvalues_and_iprs(iprs, filename_spectrum, 4, i)
        
        '''overlaps_eigenvectors_examples = spectrum.compute_overlap_eigenvectors_examples()
        
        filename_overlaps = f"spectra/overlap_eigenvector_example_N{N}_d{d}_P{P}_CloseTo{typeOfMixture}Mixture.txt"
        
        # filename_overlaps = "spectra/overlap_eigenvector_example_N"+str(N)+"_d"+str(d)+"_P"+str(P)+"_CloseTo"+str(typeOfMixture)+"Mixture.txt"
        
        ranks_of_maximum_overlap_per_example = np.argmax(overlaps_eigenvectors_examples, axis=0)
        
        eigs_of_maximum_overlap_per_example = spectrum.eigenvalues[ranks_of_maximum_overlap_per_example]
        
        spectrum.print_overlaps_eigenvectors_examples(np.max(overlaps_eigenvectors_examples, axis=0), filename_overlaps, i, ranks_of_maximum_overlap_per_example, eigs_of_maximum_overlap_per_example) ## for any pattern, prints the maximal overlap'''

        ''' decomment above to study overlap of patterns with low rank eigenvectors '''

else:
            
    sys.exit("Specify initial condition")