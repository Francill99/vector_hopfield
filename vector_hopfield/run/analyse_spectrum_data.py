import numpy as np
import sys

filename="input_for_analyse_spectrum_data.txt"

stream=open(filename, "r")

input=stream.readline().split(" ")

N=int(input[0])
d=int(input[1])
M=int(input[2])
type_minimum=input[3]

alpha = np.float64(M/N)

if type_minimum == "R":
    
    attr="farFromPatterns"
    
elif type_minimum == "C":
    
    thres_m=0.7
    attr="CloseToPattern"
    
elif type_minimum == "M":
    
    s = 3 ##order of mixture
    aS = 1.  ## asym of mixture (=1 sym mixture, != 1 asym mixture)
    
    if aS != 1:
        
        typeOfMixture = "aSym"
        
    else:
        
        typeOfMixture = "Sym"
    
    thres_m=0.7
    attr="CloseTo"+typeOfMixture+str(s)+"Mixture"

stream.close()

def get_info_final_state():
    
    filename="final_states/final_states_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+attr+".txt"
    
    stream=open(filename, "r")
    
    info = np.float64([row.split()[:4] for row in stream.readlines() if row.split()[4] == "0" and len(row.split())==8])
    
    stream.close()
    
    return info
    

def get_local_fields_magnitudes():
    
    filename="final_states/final_states_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+attr+".txt"
    
    stream=open(filename, "r")
    
    local_fields=np.float64([row.split()[:d] for row in stream.readlines() if row.split()[d+1]=="1" and len(row.split())==d+2])
    
    stream.close()
    
    return np.linalg.norm(local_fields, axis=1, keepdims=False)

def compute_histogram_local_fields_magnitudes(samples=np.zeros(0), nature_stationary_points="all", Attr=attr):
    
    local_fields_magnitudes=get_local_fields_magnitudes()
    
    if np.size(samples):
    
        nsample = np.int64(np.size(local_fields_magnitudes)/N+0.001)
        
        local_fields_magnitudes = np.reshape(local_fields_magnitudes, (nsample, N))
        
        local_fields_magnitudes = local_fields_magnitudes[samples, :]
        
        nsample = np.int64(np.size(local_fields_magnitudes)/N+0.001)
        
        local_fields_magnitudes = np.reshape(local_fields_magnitudes, (nsample*N))
        
    MIN = np.min(local_fields_magnitudes)
    MAX = np.max(local_fields_magnitudes)

    ndata = np.size(local_fields_magnitudes)

    Nbin = np.int64(np.sqrt(ndata))

    bw = (MAX-MIN)/Nbin

    tmp = np.int64((local_fields_magnitudes-MIN)/bw)

    histo=np.zeros(Nbin)

    for n in range(Nbin):

        histo[n] = np.size(tmp[tmp==n])

    histo /= (bw*ndata)

    bins = MIN + np.arange(1, Nbin+1)*bw - 0.5*bw
    
    filename="final_states/histogram_local_fields_magnitudes_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+Attr+"_"+nature_stationary_points+".txt"
    
    for i in range(Nbin):
        
        print(bins[i], histo[i], file=open(filename, "a"))
        
def compute_histogram_final_magnetisations(m, local_attr):
    
    MIN = np.min(m)
    MAX = np.max(m)
    
    ndata = np.size(m)
    Nbin = int(np.sqrt(ndata)+0.5)
    
    bw = (MAX-MIN)/Nbin
    
    tmp = np.int64((m-MIN)/bw)

    histo=np.zeros(Nbin)

    for n in range(Nbin):

        histo[n] = np.size(tmp[tmp==n])

    histo /= (bw*ndata)

    bins = MIN + np.arange(1, Nbin+1)*bw - 0.5*bw
    
    filename="final_states/histogram_final_magnetisations_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+attr+"_"+local_attr+".txt"
    
    for i in range(Nbin):
        
        print(bins[i], histo[i], file=open(filename, "a"))
        
def compute_spectral_density(eigs, nature_stationary_points="all", Attr=attr):
    
    MIN = np.min(eigs)
    MAX = np.max(eigs)

    ndata = np.size(eigs)

    Nbin = np.int64(np.sqrt(ndata))

    bw = (MAX-MIN)/Nbin

    tmp = np.int64((eigs-MIN)/bw)

    histo=np.zeros(Nbin)

    for n in range(Nbin):

        histo[n] = np.size(tmp[tmp==n])

    histo /= (bw*ndata)

    bins = MIN + np.arange(1, Nbin+1)*bw - 0.5*bw
    
    filename="spectra/spectral_density_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+Attr+"_"+nature_stationary_points+".txt"
    
    for i in range(Nbin):
        
        print(bins[i], histo[i], file=open(filename, "a"))
    
def compute_averages_info(data, nsample):
    
    m = np.mean(data)
    var = (nsample/(nsample-1))*(np.mean(data**2)-m**2)
    
    return m, np.sqrt(var/nsample)
    
def compute_averages(data, nsample):
    
    m = np.mean(data, axis=0, keepdims=False)
    var = (nsample/(nsample-1))*(np.mean(data**2, axis=0, keepdims=False)-m**2)
    
    return m, np.sqrt(var/nsample)
    
def do_full_analysis(sampless, ATTR):
    
    compute_histogram_local_fields_magnitudes(samples=sampless, Attr=ATTR)
    
    filename="spectra/spectrum_and_iprs_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+attr+".txt"
    
    stream = open(filename, "r")
    
    eigs = np.float64([row.split()[1] for row in stream.readlines()])
    
    stream.close()
    
    stream = open(filename, "r")
    
    iprs = np.float64([row.split()[3] for row in stream.readlines()])

    stream.close()
    
    nsample = np.int64(np.size(eigs)/((d-1)*N)+0.001)
    
    eigs = np.reshape(eigs, (nsample, (d-1)*N))
    iprs = np.reshape(iprs, (nsample, (d-1)*N))
    
    eigs = eigs[sampless, :]
    iprs = iprs[sampless, :]
    
    nsample = np.size(sampless)
    
    eigs = np.reshape(eigs, nsample*(d-1)*N)

    compute_spectral_density(eigs, Attr=ATTR)
    
    ## ora distingue tra minimi e selle

    eigs = np.reshape(eigs, (nsample, (d-1)*N))

    eigsp = eigs[eigs[:, 0]>0]
    npl = np.size(eigsp[:, 0])
    eigsm = eigs[eigs[:, 0]<0]
    nm = np.size(eigsm[:, 0])
    
    list_minima_samples = np.where(eigs[:, 0]>0)[0]
    
    if npl>1:
    
        compute_spectral_density(np.reshape(eigsp, (npl*(d-1)*N)), nature_stationary_points="minima", Attr=ATTR)
        compute_histogram_local_fields_magnitudes(samples=list_minima_samples, nature_stationary_points = "minima", Attr=ATTR)

        filename="spectra/spectrum_and_iprs_averages_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+ATTR+"_minima.txt"

        stream=open(filename, "a")

        eigs_avg_minima, eigs_err_minima = compute_averages(eigs[eigs[:, 0]>0], npl)
        
        iprs_avg_minima, iprs_err_minima = compute_averages(iprs[eigs[:, 0]>0], npl)

        log_iprs_avg_minima, log_iprs_err_minima = compute_averages(np.log(iprs[eigs[:, 0]>0]), npl)

        iprs_typ, iprs_typ_err = np.exp(log_iprs_avg_minima), np.exp(log_iprs_avg_minima)*log_iprs_err_minima

        for i in range((d-1)*N):
            
            print(i, eigs_avg_minima[i], eigs_err_minima[i], iprs_avg_minima[i], iprs_err_minima[i], iprs_typ[i], iprs_typ_err[i], file=stream)

        stream.close()
    
    if nm>1:

        compute_spectral_density(np.reshape(eigsm, (nm*(d-1)*N)), nature_stationary_points="saddles", Attr=ATTR)

        list_saddle_samples = np.where(eigs[:, 0]<0)[0]
    
        compute_histogram_local_fields_magnitudes(samples=list_saddle_samples, nature_stationary_points = "saddles", Attr=ATTR)
        
        filename="spectra/spectrum_and_iprs_averages_N"+str(N)+"_d"+str(d)+"_P"+str(M)+"_"+ATTR+"_saddles.txt"

        stream=open(filename, "a")

        eigs_avg_saddles, eigs_err_saddles = compute_averages(eigs[eigs[:, 0]<0], nm)

        iprs_avg_saddles, iprs_err_saddles = compute_averages(iprs[eigs[:, 0]<0], nm)
        
        log_iprs_avg_saddles, log_iprs_err_saddles = compute_averages(np.log(iprs[eigs[:, 0]<0]), nm)
        
        iprs_typ, iprs_typ_err = np.exp(log_iprs_avg_saddles), np.exp(log_iprs_avg_saddles)*log_iprs_err_saddles

        for i in range((d-1)*N):
            
            print(i, eigs_avg_saddles[i], eigs_err_saddles[i], iprs_avg_saddles[i], iprs_err_saddles[i], iprs_typ[i], iprs_typ_err[i], file=stream)

        stream.close()
    
    
info = np.transpose(get_info_final_state())
    
if type_minimum == "C" or type_minimum == "M": ## per pattern-correlated o misture meglio distinguire tra minimi che effettivamente sono molto vicini alla condizione iniziale e quelli che non lo sono
    
    if type_minimum == "C":
        
        attr1 = "LessCloseToPattern"
        
    else:
        
        attr1 = "LessCloseTo3Mixture"
    
    info_trasp = np.transpose(info)
     
    m = np.array([ciccio[0] for ciccio in info_trasp if ciccio[0]>=thres_m])
    m_all = info[0]
    m_lc = np.setdiff1d(m_all, m)
    t = np.array([ciccio[1] for ciccio in info_trasp if ciccio[0]>=thres_m])
    t_all = info[1]
    t_lc = np.setdiff1d(t_all, t)
    E = np.array([ciccio[2] for ciccio in info_trasp if ciccio[0]>thres_m])
    E_all = info[2]
    E_lc = np.setdiff1d(E_all, E)
    Samples_veryclose = np.int64(np.intersect1d(m_all, m, return_indices=True)[1]) ## assumendo che i valori di m_all sono tutti unici, funziona (probabilità che ne vengono fuori due uguali infinitesima)
    Samples = np.arange(np.size(m))
    Samples_lessclose = np.setdiff1d(Samples, Samples_veryclose)
    
    m_avg, m_err = compute_averages_info(m, np.size(m))
    t_avg, t_err = compute_averages_info(t, np.size(t))
    E_avg, E_err = compute_averages_info(E, np.size(E))
    
    ## prima analisi dei dati info del final states
    
    if np.size(Samples_lessclose)>1:
        
        m_all_avg, m_all_err = compute_averages_info(m_all, np.size(m_all))
        t_all_avg, t_all_err = compute_averages_info(t_all, np.size(t_all))
        E_all_avg, E_all_err = compute_averages_info(E_all, np.size(E_all))
        m_lc_avg, m_lc_err = compute_averages_info(m_lc, np.size(m_lc))
        t_lc_avg, t_lc_err = compute_averages_info(t_lc, np.size(t_lc))
        E_lc_avg, E_lc_err = compute_averages_info(E_lc, np.size(E_lc))

        fname = "final_states/info_avg_d"+str(d)+"_alpha"+str(alpha)+"_CloseToPattern.txt"
        
        print(N, m_avg, m_err, m_lc_avg, m_lc_err, m_all_avg, m_all_err, t_avg, t_err, t_lc_avg, t_lc_err, t_all_avg, t_all_err, E_avg/N, E_err/N, E_lc_avg/N, E_lc_err/N, E_all_avg/N, E_all_err/N, file=open(fname, "a"))
        
    else:
        
        fname = "final_states/info_avg_d"+str(d)+"_alpha"+str(alpha)+"_CloseToPattern.txt"
        
        print(N, m_avg, m_err, "X", "X", "X", "X", t_avg, t_err, "X", "X", "X", "X", E_avg, E_err, "X", "X", "X", "X", file=open(fname, "a"))

    compute_histogram_final_magnetisations(m_all, "all") ## qui mi interessa vedere tutto
    
    ## analisi completa dello spettro
    
    do_full_analysis(Samples_veryclose, attr)
    
    if np.size(Samples_lessclose)>1:
        
        do_full_analysis(Samples_lessclose, attr1)
        
else:
        
    m = np.transpose(info[0])
    t = np.transpose(info[1])
    E = np.transpose(info[2])
    Samples = np.transpose(info[3])
    
    m_avg, m_err = compute_averages_info(m, np.size(m))
    t_avg, t_err = compute_averages_info(t, np.size(t))
    E_avg, E_err = compute_averages_info(E, np.size(E))
    
    fname = "final_states/info_avg_d"+str(d)+"_alpha"+str(alpha)+"_"+attr+".txt"

    print(N, m_avg, m_err, t_avg, t_err, E_avg/N, E_err/N, file=open(fname, "a"))
    
    do_full_analysis(Samples, attr)
                
    

