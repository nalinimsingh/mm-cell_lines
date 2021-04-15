import numpy as np
from scipy.stats import spearmanr

from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


def pca_and_linreg(respmat, num, n_components = 100, alpha = 0.01, hasnans=False):
    np.random.seed(0)
    drug_n = respmat.shape[0]
    choice = np.random.choice(range(drug_n),int(0.8*drug_n),replace=False) 
    train_inds = np.zeros(drug_n, dtype=bool)
    train_inds[choice] = True
    test_inds = ~train_inds 
    
    if(hasnans):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(respmat.iloc[train_inds,-num:])
        respmat.iloc[:,-num:] = imp_mean.transform(respmat.iloc[:,-num:])
    
    pca = PCA(n_components = n_components)
    pca.fit(respmat.iloc[train_inds,-num:])
    plt.figure()
    plt.scatter(range(n_components),np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    train_feat = pca.transform(respmat.iloc[train_inds,-num:])
    test_feat = pca.transform(respmat.iloc[test_inds,-num:])
    
    train_ic50 = np.log(respmat['IC50_PUBLISHED'][train_inds])
    test_ic50 = np.log(respmat['IC50_PUBLISHED'][test_inds])
    
    reg = Lasso(alpha=alpha, normalize=False).fit(train_feat, train_ic50)
    
    print()
    print('Train R2: ', round(reg.score(train_feat, train_ic50),3))
    print('Test R2: ', round(reg.score(test_feat, test_ic50),3))
          
    print('Train Spearman Rank Corr: ', round(spearmanr(reg.predict(train_feat),train_ic50).correlation, 3))
    print('Test Spearman Rank Corr: ', round(spearmanr(reg.predict(test_feat),test_ic50).correlation, 3))
    
    print('Train Mean Absolute Error: ', round(np.mean(np.abs(reg.predict(train_feat)-train_ic50)),3))
    print('Test Mean Absolute Error: ', round(np.mean(np.abs(reg.predict(test_feat)-test_ic50)),3))
 
    return reg, train_feat, train_ic50, test_feat, test_ic50