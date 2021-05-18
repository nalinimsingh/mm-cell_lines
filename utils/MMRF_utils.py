import os, sys, glob
import numpy as np
import pandas as pd
from data import load_mmrf
from sklearn.decomposition import PCA


# Preprocess patient and genomic data in MMRF

# Written By: Qingyang Xu

# 1. Need load_mmrf function from data.py
# 2. Download the data files in ./data/



def get_train_test_valid(data_filename, ind=0, show_features=True):
    
    dset = load_mmrf(fold_span = [ind], data_dir = data_filename)
    
    train = dset[ind]['train']
    test = dset[ind]['test']
    valid = dset[ind]['valid']
    
    if show_features:
        treatment_names = train['feature_names_a']
        basline_names = train['feature_names']
        biomarker_names = train['feature_names_x']

        for key in train:
            print(key)
            arr = train[key]
            print(arr.shape)

            if 'feature_names' in key: 
                print(arr)
    
    return train, test, valid




# data can be "train", "test" or "valid"
# include the first T patient clinical visit data
def preprocess_patient_data(data, num_clin_visits=0):
    
    print('Keep first %d clinical visits.'%num_clin_visits)
    
    treatment_names = data['feature_names_a']
    basline_names = data['feature_names']
    biomarker_names = data['feature_names_x']

    df = pd.DataFrame({'pids':data['pids'], 'pfs':np.squeeze(data['ys_seq'])})
    
    df[basline_names] = data['b']

    for i in range(num_clin_visits):
        col_names = [x+str(i+1) for x in biomarker_names]
        #print(col_names)
        df[col_names] = data['x'][:,i,:]

    for i in range(num_clin_visits):
        col_names = [x+str(i+1) for x in treatment_names]
        #print(col_names)
        df[col_names] = data['a'][:,i,:]
    
    df = df[df['pfs']>0.] # remove right-censored PFS
    return df




# Format genomic data where each row is a unique MMRF patient
# If nPCA is 0, use raw genomic data for each patient
# Otherwise, run PCA on raw genomic data first
def preprocess_genomic_data(genomic_fn, nPCA=0):
    
    print('Reading %s'%genomic_fn)
    genome = pd.read_csv(genomic_fn, delimiter='\t')
    n0 = genome.shape[0]
    print('Raw data shape:')
    print(genome.shape)
    
    gene_ID = list(genome.pop('GENE_ID'))

    genome_cols = genome.columns
    genome = genome.dropna()  # may also impute the missing values
    n1 = genome.shape[0]
    
    print('%d out of %d genes have missing data '%(n0-n1,n0))
    
    cols = [col for col in genome_cols if '1_BM' in col] # keep the first bone marrow test
    genome =genome[cols]
    X = np.array(genome) 
    
    if nPCA > 0:
        print('Running PCA with %d principal components...'%(nPCA))

        pca = PCA(n_components=nPCA)
        pca_X = pca.fit_transform(np.transpose(X))

        print('Finished PCA!')
    
        df = pd.DataFrame(data=pca_X, columns=['PCA%d'%(i+1) for i in range(nPCA)])
        
    else:
        print('Using raw genomic data...')
        df = pd.DataFrame(data=np.transpose(X), columns = gene_ID)
        
    print('Created new dataframe...')
            
    df['pids'] = [p.split('_')[0]+'_'+p.split('_')[1] for p in genome.columns]
    return df



# test each module
if __name__ == '__main__':
    
    print('Get train, test, and valid sets')
    ind = 1
    data_filename = './data/cleaned_mm%d_2mos_pfs_ind.pkl'%(ind)
    train, test, valid = get_train_test_valid(data_filename, ind, show_features=True)
    
    print('\nPreprocess patient data in MMRF')
    # include the first n clinical visit data for each patient (default 0)
    train_df = preprocess_patient_data(train, num_clin_visits=3)  
    print(train_df.shape)
    
    valid_df = preprocess_patient_data(valid, num_clin_visits=3)  
    print(valid_df.shape)
    
    test_df = preprocess_patient_data(test, num_clin_visits=3)  
    print(test_df.shape)

    
    print('\nPreprocess patient genomic data in MMRF...')
    genomic_fn = './data/MMRF_CoMMpass_IA15a_E74GTF_Salmon_Gene_TPM.txt'
    genomic_df = preprocess_genomic_data(genomic_fn, nPCA=5)
    print(genomic_df.shape)
    
    print('\nMerging patient data with genomic data...')
    train_patient_genomic = train_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(train_patient_genomic.shape)
    
    valid_patient_genomic = valid_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(valid_patient_genomic.shape)
    
    test_patient_genomic = test_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(test_patient_genomic.shape)

