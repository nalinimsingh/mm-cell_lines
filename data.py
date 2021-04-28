import numpy as np
from lifelines.utils import concordance_index
import pandas as pd
import pickle    
import sys, os, warnings
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

def make_censored_oh(Y, C, K):
    """
    Helper function that zeros out Y (after it has been digitized) based on whether or not
    the patient has been censored at that particular time point. 
    """
    Yr,Cr = Y.ravel(), C.ravel()
    oh    = np.zeros((Y.shape[0], K))
    for k in range(Y.shape[0]):
        oh[k, Yr[k]]= 1.
    return oh
    
def digitize_outcomes(Y, Yvalid, Ytest, Ymax, K, method='quantiles'):
    """
    Helper function that digitizes outcomes, which means it converts a single real number
    corresponding to date of death (for example) into a sequential vector that bins intermediate
    values.
    """
    if method=='quantiles':
        K_m_2     = K-1 # correct for bin edges
        probs     = np.arange(K_m_2+1)/float(K_m_2)
        bin_edges = stats.mstats.mquantiles(Y, probs)#[0, 2./6, 4./6, 1])
        bin_edges = bin_edges.tolist()
        bin_edges+= [Ymax]
        bin_edges = bin_edges
    elif method=='uniform':
        bin_edges = np.linspace(0, Ymax, K+1)
        bin_edges = bin_edges.tolist()
    else:
        raise ValueError('bad setting for method')
    predict = []
    for k in range(len(bin_edges)-1):
        predict.append((bin_edges[k]+bin_edges[k+1])/2.)
    predict = np.array(predict)
    
    Ytr  = np.digitize(Y.astype(float).ravel(), bin_edges)-1
    Yva  = np.digitize(Yvalid.astype(float).ravel(), bin_edges)-1
    Yte  = np.digitize(Ytest.astype(float).ravel(), bin_edges)-1
    print (Ytr.max()+1, Yva.max()+1, Yte.max()+1)
    assert predict.shape[0]==K,'Expecting K categories'
    return Ytr, Yva, Yte, predict

def load_mmrf_quick(fold_span = range(5), fval=None):
    """
    Helper function to load in data tensors from pkl files.
    
    Args: 
        fold_span: list of folds that we wish to select 
        fval: path to cleaned output files
    Returns: 
        dset_rest: returns data dictionary with folds and corresponding train, test, and validation
        sets.
    """
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     fval     = os.path.join(dir_path, 'output/cleaned_mm_fold'+suffix+'.pkl')
    print( 'loading from:', fval)
    dset_rest = {}
    for foldnum in fold_span:
        fn = fval.replace('_fold',str(foldnum))
        with open(fn, 'rb') as f:
            dset = pickle.load(f)
        dset_rest[foldnum] = dset[foldnum]
    return dset_rest

def load_mmrf(fold_span = range(5), data_dir=None, digitize_K = 0, digitize_method = 'uniform', \
              subsample = False, add_syn_marker=False, restrict_markers=[], \
              window='all', data_aug=False, ablation=False, feats=[]):
    """
    Main function that loads the tensors from the stored .pkl files (which are generated by 
    running python build_mmrf_dataset.py) and returns the dataset. 
    
    Args: 
        fold_span: list of folds that we wish to select 
        suffix: string that might appear in .pkl file names (e.g. _2mos), used to define path to 
        pickle files 
        digitize_K: int that determines to what extent to "digitize" outcomes into bins 
        digitize_method: method to digitize (either 'uniform' or 'quantiles')
        subsample: bool for whether or not to subsample training set 
        add_syn_marker: bool for whether or not to include a synthetic marker that is constructed 
        from observed lab values 
        restric_markers: bool for whether or not to restrict markers to serum M protein and the 
        synthetic marker
        window: string that determines how masks are going to be constructed (e.g. only look 
        at first and second line therapies for each patient). can be "all", "first_second", 
        or "second".      
    Returns: 
        new_dset: returns data dictionary with folds and corresponding train, test, and validation
        sets.
    """
    new_dset = load_mmrf_quick(fold_span = fold_span, fval=data_dir)
    
    # Make sure we only see data up to maxT
    for fold in fold_span:
        for tvt in ['train', 'valid', 'test']:
            M      = new_dset[fold][tvt]['m']
            m_t    = ((np.flip(np.cumsum(np.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
            maxT   = m_t.sum(-1).max()
            new_dset[fold][tvt]['x'] = new_dset[fold][tvt]['x'][:,:maxT,:]
            new_dset[fold][tvt]['a'] = new_dset[fold][tvt]['a'][:,:maxT,:]
            new_dset[fold][tvt]['m'] = new_dset[fold][tvt]['m'][:,:maxT,:]
            new_dset[fold][tvt]['m_a'] = new_dset[fold][tvt]['m_a'][:,:maxT,:]
    
    if subsample: 
        # Transfer data from train to test set
        dir_path = os.path.dirname(os.path.realpath(__file__))
        fval     = os.path.join(dir_path, 'samples.pkl')
        if os.path.exists(fval):
            with open(fval,'rb') as f:
                sample_idx = pickle.load(f)
        else:
            np.random.seed(0)
            sample_idx = {}
            for fold in fold_span:
                N   = new_dset[fold]['train']['x'].shape[0]
                Ns  = int(N*0.15)
                idxshuf = np.random.permutation(N)
                sample_idx[fold] = (idxshuf[:Ns], idxshuf[Ns:])
            with open(fval,'wb') as f:
                pickle.dump(sample_idx, f)
        for fold in fold_span:
            keep, move = sample_idx[fold]
            for kk in ['a','x','m','ys_seq','ce','b','pids','m_a']:
                new_dset[fold]['test'][kk] = np.concatenate([new_dset[fold]['test'][kk], new_dset[fold]['train'][kk][move]], axis=0)
                new_dset[fold]['train'][kk]= new_dset[fold]['train'][kk][keep]
                
    if digitize_K>0:
        for fold in fold_span:
            Ytrain, Yvalid, Ytest = new_dset[fold]['train']['ys_seq'][:,0].ravel(), new_dset[fold]['valid']['ys_seq'][:,0].ravel(), new_dset[fold]['test']['ys_seq'][:,0].ravel()
            Ymax  = np.max([Ytrain.max(), Yvalid.max(), Ytest.max()])+0.1
            print ('Digitizing outcomes ymax:',Ymax)
            ytrain_bin, yvalid_bin, ytest_bin, predictions = digitize_outcomes(Ytrain, Yvalid, Ytest, Ymax, digitize_K, method=digitize_method)
            new_dset[fold]['train']['digitized_y'] = make_censored_oh(ytrain_bin, new_dset[fold]['train']['ce'].ravel(), digitize_K)
            new_dset[fold]['valid']['digitized_y'] = make_censored_oh(yvalid_bin, new_dset[fold]['valid']['ce'].ravel(), digitize_K)
            new_dset[fold]['test']['digitized_y']  = make_censored_oh(ytest_bin,  new_dset[fold]['test']['ce'].ravel(), digitize_K)
            new_dset[fold]['train']['prediction'] = predictions
            new_dset[fold]['valid']['prediction'] = predictions
            new_dset[fold]['test']['prediction']  = predictions
    for fold in fold_span:
        for k in ['train','valid','test']:
            m    = (new_dset[fold][k]['m_a'].sum(-1)>0.)*1.
            mask = (m[:,::-1].cumsum(1)[:,::-1]>0)*1.
            lot  = new_dset[fold][k]['a'][...,-1]
            lot[:,0]    = 1.
            lot[lot==0] = np.nan
            df = pd.DataFrame(lot)
            df.fillna(method='ffill', axis=1, inplace=True) # forward fill
            lot = df.values
            lot = lot*mask
            lot[lot>=3] = 3.
            lot_oh      = np.zeros(lot.shape+(4,))
            for i in range(lot.shape[0]):
                for j in range(lot.shape[1]):
                    lot_oh[i,j,lot[i,j].astype(int)] = 1
            lot_oh      = lot_oh[...,1:]
            time_val    = np.ones_like(lot_oh[:,:,[-1]])
            time_val    = np.cumsum(time_val, 1)*0.1
            time_val    = (lot_oh.cumsum(1)*lot_oh*0.1).sum(-1,keepdims=True)
            new_dset[fold][k]['a'] = np.concatenate([time_val, new_dset[fold][k]['a'][...,:-1], lot_oh], -1)
            new_dset[fold][k]['feature_names_a'] = np.array(['local_clock']+new_dset[fold][k]['feature_names_a'].tolist()[:-1]+['line1','line2','line3plus'])
            
            if add_syn_marker: # synthetic marker is sum of two major Igs based on myeloma type
                x    = new_dset[fold][k]['x']
                m    = new_dset[fold][k]['m']
                b    = new_dset[fold][k]['b']
                b_names = new_dset[fold][k]['feature_names'].tolist()
                x_names = new_dset[fold][k]['feature_names_x'].tolist()
                new_x   = np.zeros((x.shape[0],x.shape[1],x.shape[2]+1))
                new_m   = np.ones((m.shape[0],m.shape[1],m.shape[2]+1))
                for i in range(x.shape[0]): 
                    tseq = np.zeros((x.shape[1],)) 
                    mhseq = np.ones((x.shape[1],)); mlseq = np.ones((x.shape[1],))
                    if b[i,b_names.index('igg_type')] == 1.: 
                        tseq += x[i,:,x_names.index('serum_igg')]
                        mhseq = m[i,:,x_names.index('serum_igg')]
                    elif b[i,b_names.index('iga_type')] == 1.: 
                        tseq += x[i,:,x_names.index('serum_iga')]
                        mhseq = m[i,:,x_names.index('serum_iga')]
                    elif b[i,b_names.index('igm_type')] == 1.: 
                        tseq += x[i,:,x_names.index('serum_igm')]
                        mhseq = m[i,:,x_names.index('serum_igm')]
                    
                    if b[i,b_names.index('kappa_type')] == 1.: 
                        tseq += x[i,:,x_names.index('serum_kappa')]
                        mlseq = m[i,:,x_names.index('serum_kappa')]
                    elif b[i,b_names.index('lambda_type')] == 1.: 
                        tseq += x[i,:,x_names.index('serum_lambda')] 
                        mlseq = m[i,:,x_names.index('serum_lambda')]
        
                    tseq = tseq[:,np.newaxis]
                    new_x[i] = np.concatenate((new_dset[fold][k]['x'][i,:,:], tseq), axis=-1)
                    mseq     = np.ones((x.shape[1],1)) # change this line to make sure missingness pattern is correct (only 0 if both are missing)
#                     mseq = (mhseq.astype(int) | mlseq.astype(int)).astype(float)[:,np.newaxis]
                    new_m[i] = np.concatenate((new_dset[fold][k]['m'][i,:,:], mseq), axis=-1)
                new_dset[fold][k]['x'] = new_x; new_dset[fold][k]['m'] = new_m
                new_dset[fold][k]['feature_names_x'] = np.array(x_names + ['syn_marker'])
                print(f'adding synthetic marker in fold {fold}, set {k}...')
                print(f"new shape of X: {new_dset[fold][k]['x'].shape}")
                print(f"new shape of M: {new_dset[fold][k]['m'].shape}")
    
    if len(restrict_markers) != 0: 
        for fold in fold_span: 
            for k in ['train', 'valid', 'test']: 
                x_names = new_dset[fold][k]['feature_names_x'].tolist()
                fs = []; ms = []
                for name in restrict_markers: 
                    f = new_dset[fold][k]['x'][...,x_names.index(name)][:,:,np.newaxis]
                    m = new_dset[fold][k]['m'][...,x_names.index(name)][:,:,np.newaxis]
                    fs.append(f); ms.append(m)
                new_dset[fold][k]['x'] = np.concatenate(tuple(fs),axis=-1)
                new_dset[fold][k]['m'] = np.concatenate(tuple(ms),axis=-1)
                new_dset[fold][k]['feature_names_x'] = np.array(restrict_markers)
                print(f'restricting longitudinal markers in fold {fold}, set {k}...')
                print(f"new shape of X: {new_dset[fold][k]['x'].shape}")
                print(f"new shape of M: {new_dset[fold][k]['m'].shape}")
    
    if window == 'first_second':
        d = set()
        for fold in fold_span: 
            for k in ['train', 'valid', 'test']: 
                new_m = np.copy(new_dset[fold][k]['m'])
                a = new_dset[fold][k]['a']
                for pt in range(new_m.shape[0]): 
                    sec_idxs= np.where(a[pt,:,-2] == 1.)[0]
                    if len(sec_idxs) == 0: 
                        d.add(new_dset[fold][k]['pids'][pt])
                        # print(a[pt,:,:])
                        sec_end = np.min([np.max(np.where(a[pt,:,-3] == 1.)[0])+3,new_m.shape[1]])
                    else: 
                        sec_end = np.min([np.max(np.where(a[pt,:,-2] == 1.)[0])+3,new_m.shape[1]])
                    new_m[pt,sec_end:,:] = 0.
                new_dset[fold][k]['m'] = new_m
        #print(d)
    elif window == 'second':
        pass # implement mask alteration or altering dataset for restricting line of therapy
    
    if data_aug: 
        for fold in fold_span:
            augment_data(new_dset[fold], digitize_K=digitize_K)
    
    if ablation:
        ablate_idxs = {
            'none': (0,0),
            'none_trt': (0,2),
            'all': (0,new_dset[fold_span[0]]['train']['b'].shape[-1]), 
            'demog': (1,3),
            'iss': (0,4),
            'pc': (0,9),
            'hc': (0,10),
            'igg': (0,13), 
            'lc': (0,2),
            'comb_trts': (0,7),
            'asct': (0,8),
            'bor': (0,2),
            'car': (0,2), 
            'cyc': (0,3),
            'dex': (0,4),
            'len': (0,5),
            'lines': (0,new_dset[fold_span[0]]['train']['a'].shape[-1])
        }
        assert len(feats) != 0, 'Need to pass in the ablation params in feats argument'
        for fold in fold_span: 
            for k in ['train', 'valid', 'test']: 
                include_baseline, include_treatment = feats[0], feats[1]
                bsidx, beidx = ablate_idxs[include_baseline]
                tsidx, teidx = ablate_idxs[include_treatment]
                
                new_dset[fold][k]['b'] = new_dset[fold][k]['b'][:,bsidx:beidx+1]
                new_dset[fold][k]['a'] = new_dset[fold][k]['a'][...,tsidx:teidx+1]
                if include_treatment == 'none_trt': 
                    new_dset[fold][k]['a'][...] = 0.
                elif include_treatment == 'lc': 
                    new_dset[fold][k]['a'][:,:,1:] = 0.
                elif include_treatment == 'bor': 
                    new_dset[fold][k]['a'][:,:,2] = 0.
                new_dset[fold][k]['feature_names'] = new_dset[fold][k]['feature_names'][bsidx:beidx+1]
                new_dset[fold][k]['feature_names_a'] = new_dset[fold][k]['feature_names_a'][tsidx:teidx+1]
                
    return new_dset

def augment_data(dset_fold, mult=25, digitize_K=0): 
    B = dset_fold['train']['b']; X = dset_fold['train']['x']
    M = dset_fold['train']['m']; CE= dset_fold['train']['ce']
    A = dset_fold['train']['a']; Y = dset_fold['train']['ys_seq']
    
    nsamples = mult*B.shape[0]
    if digitize_K > 0: 
        Ydig = dset_fold['train']['digitized_y']
        Ydigs = np.zeros((nsamples,Ydig.shape[1]))
        
    Bs = np.zeros((nsamples,B.shape[1]))
    Xs = np.zeros((nsamples,X.shape[1],X.shape[2]))
    As = np.zeros((nsamples,A.shape[1],A.shape[2]))
    Ms = np.zeros((nsamples,M.shape[1],M.shape[2]))
    if len(Y.shape) == 1: 
        Ys = np.zeros((nsamples,))
    else: 
        Ys = np.zeros((nsamples,Y.shape[1]))
    CEs = np.zeros((nsamples,CE.shape[1]))
    
    for i in range(mult): 
        As[i*A.shape[0]:(i+1)*A.shape[0]] = A
        Bs[i*B.shape[0]:(i+1)*B.shape[0]] = B
        Ms[i*M.shape[0]:(i+1)*M.shape[0]] = M
        Ys[i*Y.shape[0]:(i+1)*Y.shape[0]] = Y
        if digitize_K > 0: 
            Ydigs[i*Ydig.shape[0]:(i+1)*Ydig.shape[0]]= Ydig
        CEs[i*CE.shape[0]:(i+1)*CE.shape[0]] = CE
        
        p = np.random.uniform(0,1)
        if i == 0: 
            Xs[i*X.shape[0]:(i+1)*X.shape[0]] = X
        elif p <= 0.5: 
            shift_factor = np.random.uniform(-4,4,size=X.shape[0])[:,None,None]
            Xs[i*X.shape[0]:(i+1)*X.shape[0]] = X+shift_factor            
        else: 
            scale_factor = np.random.uniform(1,3,size=X.shape[0])[:,None,None]
            Xs[i*X.shape[0]:(i+1)*X.shape[0]] = X*scale_factor
    
    dset_fold['train']['b'] = Bs; dset_fold['train']['x'] = Xs 
    dset_fold['train']['m'] = Ms; dset_fold['train']['ce'] = CEs
    dset_fold['train']['a'] = As; dset_fold['train']['ys_seq']= Ys
    
    if digitize_K > 0: 
        dset_fold['train']['digitized_y'] = Ydigs        

def get_te_matrix(): 
    ''' 
        5x16 matrix that contains direction of treatment effect on subset of lab features. 
        - 'Bor': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: 0, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        (http://chemocare.com/chemotherapy/drug-info/bortezomib.aspx, 
        https://www.ncbi.nlm.nih.gov/pubmed/20061695 [renal], 
        https://www.nature.com/articles/s41598-017-13486-x [renal], 
        https://clinicaltrials.gov/ct2/show/NCT00972959 [calcium])
        
        - 'Car': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: +1, Hb:-1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        (https://www.rxlist.com/kyprolis-side-effects-drug-center.htm [side effects])
        
        - 'Cyc': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: 0, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        [https://www.rxlist.com/cytoxan-side-effects-drug-center.htm]
        
        - 'Dex': PMN: -1, alb: 0, BUN: 0, Ca: -1, Crt: 0, Glc: +1, Hb: +1, Kappa: 0, 
        M-prot: 0, Plt: +1, TotProt: 0, WBC: -1, IgA: 0, IgG: 0, IgM: 0, Lambda: 0 
        [https://dm5migu4zj3pb.cloudfront.net/manuscripts/108000/108231/JCI75108231.pdf], 
        
        - 'Len': PMN: -1, alb: 0, BUN: +1, Ca: -1, Crt: +1, Glc: -1, Hb: -1, Kappa: -1, 
        M-prot: -1, Plt: -1, TotProt: -1, WBC: -1, IgA: -1, IgG: -1, IgM: -1, Lambda: -1 
        [https://www.revlimid.com/mm-patient/about-revlimid/what-are-the-possible-side-effects/#common, 
        https://themmrf.org/multiple-myeloma/treatment-options/standard-treatments/revlimid/, 
        https://www.webmd.com/drugs/2/drug-94831/revlimid-oral/details/list-sideeffects]
        
        order of columns: array(['cbc_abs_neut', 'chem_albumin', 'chem_bun', 'chem_calcium',
           'chem_creatinine', 'chem_glucose', 'cbc_hemoglobin', 'serum_kappa',
           'serum_m_protein', 'cbc_platelet', 'chem_totprot', 'cbc_wbc',
           'serum_iga', 'serum_igg', 'serum_igm', 'serum_lambda'],
           dtype='<U15')
        order of rows: 'Bor', 'Car', 'Cyc', 'Dex', 'Len'
    '''
    te_matrix = np.array([[-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 
                          [-1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0], 
                          [-1, 0, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    
    return te_matrix

if __name__=='__main__':
    print ('loading dataset quickly')
    dset = load_mmrf(fold_span=range(5), digitize_K = 20, digitize_method = 'uniform', suffix='_2mos_tr', restrict_markers=[], add_syn_marker=True, window='first_second')
    for k in dset[1]['train']:
        print (k, dset[1]['train'][k].shape)
