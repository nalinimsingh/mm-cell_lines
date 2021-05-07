# ## Transfer learning from CCLE to MMRF
# 
# Last Modified: 05/06/2021
# 
# Written By: Sumi Thakur and Qingyang Xu
# 
# - Pretrain on CCLE cell line data to predict IC50
# 
# - Transfer to patient RNA-seq data to predict PFS
# 
# References
# 
# - Download patient genomic data (e.g. `MMRF_CoMMpass_IA15a_CNA_Exome_PerGene_LargestSegment.txt`)
# 
# https://research.themmrf.org/
# 
# - Download DevMap cell line data (e.g. `CCLE_expression.csv`)
# 
# https://depmap.org/portal/download/



import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from kerastuner import HyperModel
from keras import models, layers,regularizers
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners.bayesian import BayesianOptimization
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import neural_network

from sklearn import metrics as skmetrics
from sklearn.decomposition import PCA

import MMRF_utils


# ## 1. Pretrain on CCLE IC50 data

def normalize_features(train, test):
    
    print('Normalizing input features...')
    nsample, nfeature = train.shape
    
    assert nfeature == test.shape[1]

    for i in range(nfeature):
        mu = np.mean(train[:,i])
        sigma = np.std(train[:,i])
        train[:,i] -= mu
        test[:,i] -= mu
        
        if sigma > 0: # some genes may have zero variance
            train[:,i] /= sigma
            test[:,i] /= sigma
    
    return train, test



def preprocess_CCLE(ccle_exp, sample_info, sanger_dose_response, mapping):
    
    print('Preprocessing CCLE data...')
    
    ccle_exp = ccle_exp.rename(columns={'Unnamed: 0':'DepMap_ID'})
    cols_to_keep = ['DepMap_ID'] + list(pd.unique(mapping.HGNC_ID))
    ccle_chosen = ccle_exp[cols_to_keep]

    sanger_dose_response_filt = sanger_dose_response[(sanger_dose_response.DRUG_NAME.str.contains("BORTEZOMIB"))
                                                | (sanger_dose_response.DRUG_NAME.str.contains("LENALIDOMIDE"))
                                                ]

    sanger_dose_response_filt = sanger_dose_response_filt.rename(columns={'ARXSPAN_ID':'DepMap_ID'})
    
    merged_df = sanger_dose_response_filt.merge(ccle_chosen, on= 'DepMap_ID')[list(ccle_chosen.columns)+['DRUG_NAME','IC50_PUBLISHED']]
    merged_df['log(IC_50)'] = merged_df.IC50_PUBLISHED.apply(np.log10)
    
    merged_df_bort = merged_df[merged_df.DRUG_NAME == 'BORTEZOMIB'].drop_duplicates()
    merged_df_lenal = merged_df[merged_df.DRUG_NAME == 'LENALIDOMIDE'].drop_duplicates()
    bort_labels = merged_df_bort['log(IC_50)']
    lenal_labels = merged_df_lenal['log(IC_50)']
    bort_data = merged_df_bort.drop(columns = ['DepMap_ID','IC50_PUBLISHED','DRUG_NAME','log(IC_50)'])
    lenal_data = merged_df_lenal.drop(columns = ['DepMap_ID','IC50_PUBLISHED','DRUG_NAME','log(IC_50)'])
    
    return bort_data, bort_labels, lenal_data, lenal_labels



def split_train_test_CCLE(X, y, normalize=True, nPCA=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_train, X_test = normalize_features(X_train, X_test)
    
    pca = None
    # try running PCA on raw RNAseq data
    if nPCA > 0:
        print('Running PCA-%d on gene expressions...'%nPCA)
        pca = PCA(n_components=nPCA)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    
    assert X_train.shape[0] == y_train.shape[0]
    
    assert X_test.shape[0] == y_test.shape[0]
    
    print('Training set shape:')
    print(X_train.shape)
    
    print('Test set shape:')
    print(X_test.shape)
    
    return X_train, y_train, X_test, y_test, pca



def linear_regression_IC50(X_train, y_train, X_test, y_test):
    print('Running linear regression to predict IC50...')
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    R2 = r2_score(y_test, y_pred)
    print('R2: %0.2f'%R2)

    rho, p = pearsonr(y_test, y_pred)
    print('Correlation: %0.2f'%rho)
    print('p-value: %0.5f'%p)



class MLPModel(tf.keras.Model):

    def __init__(self,hidden_layers):
        super(MLPModel, self).__init__()
        
        assert len(hidden_layers) > 0
        self.dnn = []
        self.hidden_layers = hidden_layers
        for h in hidden_layers:
            self.dnn.append(tf.keras.layers.Dense(h, activation=tf.nn.relu, bias_initializer='random_normal'))
        self.linear = tf.keras.layers.Dense(5, bias_initializer='random_normal')        
        self.final = tf.keras.layers.Dense(1, bias_initializer='random_normal')
        self.dropout = tf.keras.layers.Dropout(0.2)
    
  
    def call(self, inputs, training=False):
        x = inputs
        for nn in self.dnn:
            x = nn(x)
            if training: 
                x = self.dropout(x)
        x = self.linear(x)
        x = self.final(x)
        return x
    
    def encode(self, inputs):
        x = inputs
        for nn in self.dnn:
            x = nn(x)
        x = self.linear(x)
        return x



def train_MLP_model(X_train, y_train, X_test, y_test, params):
    
    print('Training MLP model...')
    
    print(params)
    
    model = MLPModel(params['hidden_layers'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss="mse", metrics=["mae"])

    R2_his = []
    rho_his = []
    p_his = []

    train_loss = []
    val_loss = []

    # total training epochs = niter * nepoch
    for i in range(params['niter']):

        print('Epoch %d'%i)

        # record losses after training nepochs
        history = model.fit(X_train, y_train, epochs=params['nepoch'], validation_data=(X_test, y_test),
                            batch_size=params['batch_size'])

        train_loss += history.history['loss']
        val_loss += history.history['val_loss']

        y_pred = np.squeeze(model.predict(X_test))

        rho, p = pearsonr(y_test, y_pred)
        rho_his.append(rho)
        p_his.append(p)

        R2 = r2_score(y_test, y_pred)
        R2_his.append(R2)
        
    return model, R2_his, rho_his, p_his, train_loss, val_loss



def plot_R2_correlation(R2_his, rho_his):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$R^2$', color=color)
    ax1.plot(R2_his, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'$\rho$', color=color)  # we already handled the x-label with ax1
    ax2.plot(rho_his, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(r'$R^2$ and $\rho$ during training')
    plt.show()



def plot_losses(train_loss,val_loss):

    plt.yscale('log')
    plt.plot(train_loss[1:], label='Train Loss')
    plt.plot(val_loss[1:], label='Valid Loss')
    plt.legend()
    plt.title('Train vs. Valid Losses during Training')
    plt.xlabel('Epoch')
    plt.show()

    

# ## 2. Transfer to MMRF patient data

def preprocess_MMRF_genomic(genomic_fn, CCLE_cols, mapping):

    print('\nPreprocess patient genomic data in MMRF...')
    genomic_df = MMRF_utils.preprocess_genomic_data(genomic_fn, nPCA=0)

    mapping_dict = {}

    # create mapping dict from Ensembl to HGNC
    for i in range(mapping.shape[0]):
        eid = mapping['Ensembl_ID'].iloc[i]
        hid = mapping['HGNC_ID'].iloc[i]
        mapping_dict[eid] = hid
        
    # average over columns with the same Ensembl ID
    gene_df = genomic_df[mapping['Ensembl_ID']]
    pids = genomic_df['pids']

    HGNC_cols = [mapping_dict[eid] for eid in gene_df.columns]
    gene_df.columns = HGNC_cols
    # average over columns with the same HGNC name
    gene_df = gene_df.groupby(by=gene_df.columns, axis=1).mean()
    
    gene_df = gene_df[CCLE_cols]
    
    # normalize each gene
    for col in gene_df.columns:
        mu = gene_df[col].mean()
        sigma = gene_df[col].std()
        gene_df[col] -= mu
        if sigma>0: gene_df[col] /= sigma
    
    return gene_df, pids


# ## 3. Comparing AUC of three approaches

def predict_PFS(clf, model_name, train_X, train_y, test_X, test_y):
    
    print('Fitting model %s...'%model_name)
    clf.fit(train_X, train_y)

    # predict PoS in test set
    print('Predicting on test set...')
    y_pred = clf.predict_proba(test_X)[:,1]

    auc = skmetrics.roc_auc_score(test_y, y_pred)
    f1 = skmetrics.f1_score(test_y, y_pred.round())
    pres = skmetrics.precision_score(test_y, y_pred.round())
    rec = skmetrics.recall_score(test_y, y_pred.round())
    acc = skmetrics.accuracy_score(test_y, y_pred.round())

    print('Accuracy: %0.2f'%acc)
    print('Precision: %0.2f'%pres)
    print('Recall: %0.2f'%rec)
    print('F1 score: %0.2f'%f1)
    print('AUC: %0.2f'%auc)
    print('')


# ### 3.1 Only use patient data

def predict_pfs_patient_only(train_df,valid_df,test_df, clf, model_name='',drug='Bor'):

    print('Predicting PFS using patient clinical data only...')
    
    assert drug == 'Bor' or drug == 'Len'
    
    print('Predicting patients with %s as first line treatment...'%drug)
    
    # no genomic data
    train = pd.concat([train_df,valid_df])
    train = train[(train[drug+'1']==1)]
    print(train.shape)

    test = test_df[test_df[drug+'1']==1]
    print(test.shape)
    
    pfs_thresh = 12
    train_y = np.array(train.pop('pfs'))
    train_y = np.array([int(pfs>pfs_thresh) for pfs in train_y])
    p1 = train.pop('pids')
    train_X = np.array(train)

    test_y = np.array(test.pop('pfs'))
    test_y = np.array([int(pfs>pfs_thresh) for pfs in test_y])
    p0 = test.pop('pids')
    test_X = np.array(test)
    
    predict_PFS(clf, model_name, train_X, train_y, test_X, test_y)



# ### 3.2 PCA on raw RNA

def predict_pfs_patient_PCA(genomic_fn, train_df,valid_df,test_df, clf, model_name='',drug='Bor',nPCA=20):
    
    print('Predicting PFS using patient clinical data and PCA on RNAseq...')
    
    assert drug == 'Bor' or drug == 'Len'
    assert nPCA > 0
    
    print('Predicting patients with %s as first line treatment...'%drug)
    
    print('\nPreprocess patient genomic data in MMRF...')
    
    genomic_df = MMRF_utils.preprocess_genomic_data(genomic_fn, nPCA=nPCA)
    print(genomic_df.shape)
    
    print('\nMerging patient data with genomic data...')
    train_patient_genomic = train_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(train_patient_genomic.shape)

    valid_patient_genomic = valid_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(valid_patient_genomic.shape)

    test_patient_genomic = test_df.merge(genomic_df, left_on='pids', right_on='pids')
    print(test_patient_genomic.shape)
    
    train = pd.concat([train_patient_genomic,valid_patient_genomic])
    train = train[(train[drug+'1']==1)]
    print(train.shape)

    test = test_patient_genomic[test_patient_genomic[drug+'1']==1]
    print(test.shape)
    
    pfs_thresh = 12
    train_y = np.array(train.pop('pfs'))
    train_y = np.array([int(pfs>pfs_thresh) for pfs in train_y])
    p1 = train.pop('pids')
    train_X = np.array(train)

    test_y = np.array(test.pop('pfs'))
    test_y = np.array([int(pfs>pfs_thresh) for pfs in test_y])
    p0 = test.pop('pids')
    test_X = np.array(test)
    
    predict_PFS(clf, model_name, train_X, train_y, test_X, test_y)



# ### 3.3 Transfer from CCLE

def predict_pfs_patient_CCLE(enc_gene_df, train_df,valid_df,test_df, clf, model_name='',drug='Bor'):

    print('\nMerging patient data with genomic data...')
    train_patient_genomic = train_df.merge(enc_gene_df, left_on='pids', right_on='pids')
    print(train_patient_genomic.shape)

    valid_patient_genomic = valid_df.merge(enc_gene_df, left_on='pids', right_on='pids')
    print(valid_patient_genomic.shape)

    test_patient_genomic = test_df.merge(enc_gene_df, left_on='pids', right_on='pids')
    print(test_patient_genomic.shape)
    
    train = pd.concat([train_patient_genomic,valid_patient_genomic])
    train = train[train[drug+'1']==1]
    print(train.shape)

    test = test_patient_genomic[test_patient_genomic[drug+'1']==1]
    print(test.shape)
    
    pfs_thresh = 12
    train_y = np.array(train.pop('pfs'))
    train_y = np.array([int(pfs>pfs_thresh) for pfs in train_y])
    p1 = train.pop('pids')
    train_X = np.array(train)

    test_y = np.array(test.pop('pfs'))
    test_y = np.array([int(pfs>pfs_thresh) for pfs in test_y])
    p0 = test.pop('pids')
    test_X = np.array(test)
    
    predict_PFS(clf, model_name, train_X, train_y, test_X, test_y)


if __name__ == '__main__':
    
    print('Testing transfer learning pipeline from CCLE to MMRF...')
    
    print('Reading data files...')
    
    data_dir = './data/' # Enter the path to your folder here

    # have the following files in your data folder
    ccle_exp = pd.read_csv(data_dir+"CCLE_expression.csv")
    sample_info = pd.read_csv(data_dir+"sample_info.csv")
    sanger_dose_response = pd.read_csv(data_dir+"sanger-dose-response.csv")
    mapping = pd.read_csv(data_dir+'Ensembl_HGNC_map_042421.csv')
    genomic_fn = data_dir+'MMRF_CoMMpass_IA15a_E74GTF_Salmon_Gene_TPM.txt'
    
    ind = 1
    mmrf_filename = data_dir+'cleaned_mm%d_2mos_pfs_ind.pkl'%(ind)

    print('\nPart I. Pretraining on CCLE IC50 data')
    
    bort_data, bort_labels, lenal_data, lenal_labels = preprocess_CCLE(ccle_exp, sample_info, sanger_dose_response, mapping)

    X_train, y_train, X_test, y_test, pca = split_train_test_CCLE(bort_data, bort_labels, normalize=True, nPCA=100)

    #linear_regression_IC50(X_train, y_train, X_test, y_test) # linear regression baseline
    
    MLP_params = {'niter':30, 'nepoch':10,'batch_size':32,'hidden_layers':[50,20,10],'lr':1e-3}
    model, R2_his, rho_his, p_his, train_loss, val_loss = train_MLP_model(X_train, y_train, X_test, y_test, MLP_params)
    plot_R2_correlation(R2_his, rho_his)
    plot_losses(train_loss,val_loss)
    print('')

    ############################################################################
    
    print('\nPart II. Preprocess MMRF genomic data')

    # run PCA on MMRF genomic data and use IC50 model in Part I to encode
    gene_df,pids = preprocess_MMRF_genomic(genomic_fn, bort_data.columns, mapping)
    pca_gene = pca.transform(gene_df)
    enc_gene = model.encode(np.array(pca_gene)).numpy()

    enc_gene_df = pd.DataFrame(data=enc_gene, columns=['Enc%d'%(i+1) for i in range(enc_gene.shape[1])])
    enc_gene_df['pids'] = pids
    print('')
    
    ############################################################################
    
    print('\nPart III. Predict PFS on MMRF patient data')
    train, test, valid = MMRF_utils.get_train_test_valid(mmrf_filename, ind, show_features=True)

    print('\nPreprocess patient data in MMRF')
    # include the first n clinical visit data for each patient (default 0)
    train_df = MMRF_utils.preprocess_patient_data(train, num_clin_visits=3)  
    print(train_df.shape)

    valid_df = MMRF_utils.preprocess_patient_data(valid, num_clin_visits=3)  
    print(valid_df.shape)

    test_df = MMRF_utils.preprocess_patient_data(test, num_clin_visits=3)  
    print(test_df.shape)

    # Model 1: patient data only
    clf = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=0)
    predict_pfs_patient_only(train_df, valid_df, test_df, clf, model_name='RF', drug='Bor')
    
    # Model 2: patient data with PCA on MMRF RNAseq
    clf = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=0)
    predict_pfs_patient_PCA(genomic_fn, train_df, valid_df, test_df, clf, model_name='RF',drug='Bor',nPCA=50)
    
    # Model 3: Transfer from CCLE to MMRF
    clf = RandomForestClassifier(n_estimators=1000, max_depth=50, random_state=0)
    predict_pfs_patient_CCLE(enc_gene_df, train_df, valid_df, test_df, clf, model_name='RF',drug='Bor')
    
    print('Work complete!')

