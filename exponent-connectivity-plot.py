# study of the exponent gamma as a function of graph connectivity
# case A: the nearest number of neighbors is changing
# case B: random edges are added to the graph
# the base graph is two-dimenisonal random graph (GMM)

import numpy as np
import pickle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
import os
from sklearn.linear_model import LinearRegression
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.neighbors import kneighbors_graph

n = 10000 # graph size
beta = 0.3 #infection rate
alpha = 0.1 #recovery rate
I0 = 0.001 # initial infectioin rate
eta = 0.01 # susceptible rate

random_init_infection = 1


# case A: varying the nearest neighbor connection m
ms = [4,5,6,8,10]
randm = 0.000

plt.figure()
colors=['C0','C1','C2','C3','C4']

for (i,m) in enumerate(ms):
    pickle_f = open('data/GMM_n_%i_m_%i_randm_%.3f_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.pkl' %(n,m,randm,beta,alpha,eta,random_init_infection),"rb")
    data_dict = pickle.load(pickle_f)
    pickle_f.close()

    I  = data_dict['I']
    S  = data_dict['S']
    dI = data_dict['dI']
    R  = data_dict['R']

    log_S  = np.log(S) 
    log_I  = np.log(I) 
    log_dI = np.log(dI)
    idx = np.isfinite(log_I)*np.isfinite(log_dI)
    
    X = np.vstack((log_I[idx],log_S[idx])).T
    Y = log_dI[idx]
    reg = LinearRegression().fit(X,Y)
    gamma = reg.coef_[0]
    kappa = reg.coef_[1]
    beta_cf = reg.intercept_
    
    plt.loglog( I, dI, ls='none', marker='o', ms=4, alpha=0.7, color=colors[i] )
    plt.loglog( I[idx], np.exp(beta_cf)*(I[idx]**gamma)*(S[idx]**kappa), label=(r'$k =%i,$' + '  ' + r'$\gamma=%0.2f$') %(m,gamma), ls='--', lw=2, color=colors[i],alpha=0.8)

plt.legend(fontsize=12,ncol=1)
plt.grid()
plt.xlabel(r'$I$',fontsize=15)
plt.ylabel(r'$\Delta I$',fontsize=15,rotation=0)


plt.savefig('figs/exponent_ms.png')

# case B: adding random edges
m = 4
random_edges_pr = [0.000,0.02,0.05,0.1]

plt.figure()
colors=['C0','C1','C2','C3','C4']

for (i,randm) in enumerate(random_edges_pr):

    pickle_f = open('data/GMM_n_%i_m_%i_randm_%.3f_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.pkl' %(n,m,randm,beta,alpha,eta,random_init_infection),"rb")
    data_dict = pickle.load(pickle_f)
    pickle_f.close()

    I  = data_dict['I']
    S  = data_dict['S']
    dI = data_dict['dI']
    R  = data_dict['R']


    log_S  = np.log(S) 
    log_I  = np.log(I) 
    log_dI = np.log(dI)
    idx = np.isfinite(log_I)*np.isfinite(log_dI)
    
    X = np.vstack((log_I[idx],log_S[idx])).T
    Y = log_dI[idx]
    reg = LinearRegression().fit(X,Y)
    gamma = reg.coef_[0]
    kappa = reg.coef_[1]
    beta_cf = reg.intercept_
    print(gamma)
    
    plt.loglog( I, dI, ls='none', marker='o', ms=4, alpha=0.7, color=colors[i] )
    plt.loglog( I[idx], np.exp(beta_cf)*(I[idx]**gamma)*(S[idx]**kappa), label=(r'$m=%i,$' + ' ' + r'$\gamma=%0.2f$') %(randm*n,gamma), ls='--', lw=2, color=colors[i],alpha=0.8)
    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r'$I$',fontsize=15)
    plt.ylabel(r'$\Delta I$',fontsize=15,rotation=0)


plt.savefig('figs/exponent_random_edges.png')

plt.show()
