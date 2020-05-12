'''
    infection spread over a random graph model. 
    Nodes are samples from a mixture of Gaussian model
    each node is connected to k closest neighbors
'''
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
import os
import pickle
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.neighbors import kneighbors_graph

def sample_GMM(mus, Sigmas, N):
    indexes = np.random.randint(0,3,N)
    X = np.random.randn(N,2)
    for i in range(N):
        X[i,:] = np.dot(Sigmas[indexes[i]],X[i,:]) + mus[indexes[i]] 

    return X

def create_GMM_adjacency(mus,Sigmas,n,pr):
    X_nodes = sample_GMM(mus,Sigmas,n)
    A = kneighbors_graph(X_nodes, m, mode='connectivity', include_self=False)
    A = A.toarray()
    A_rand = np.array(np.random.rand(n,n)< pr, int) # adding the random edges 
    for i in range(n):
        A_rand[i,i] = 0

    A += A_rand
    A = A+A.T # making the matrix symmetric
    A[A>0.2] = 1.0 # nonzero entries are equal to one
    return A

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=10000, help='graph size')
parser.add_argument('--m', type=int, default=4, help='each node is connected to m nearest neighbors')
parser.add_argument('--randm', type=float, default=0.00, help='average number of neighbors added randomly to each node')
parser.add_argument('--nt', type=int, default=50, help='time-steps')
parser.add_argument('--beta', type=float, default=0.3, help='infection rate')
parser.add_argument('--alpha', type=float, default=0.1, help='recovery rate')
parser.add_argument('--I0', type=float, default=0.001, help='initial infectioin probability')
parser.add_argument('--eta', type=float, default=0.01, help='recovery to susceptible rate')
parser.add_argument('--random_init_infection', type=int, default=1, help='0 if the initial infected population is fixed. 1 if the initial infected population is selected randomly')

opt = parser.parse_args()
print(opt)

n = opt.n   # graph size
m = opt.m       # ecch node is connected to m nearest neighbors 
nt = opt.nt    # time-steps
beta = opt.beta  # infection rate
alpha = opt.alpha #recovery rate
I0 = opt.I0  # initial infectioin probability
eta = opt.eta  # redcovery to susceptible rate

randm = opt.randm # average number of neighbors added randomly to each node
pr = randm/n # probability of a new random edge

random_init_infection = opt.random_init_infection # 0 if the initial infected population is fixed. 1 if the initial infected population is selected randomly


# GMM model parameters
mu1 = np.array([0,0])
mu2 = np.array([5,1])
mu3 = np.array([1,5])
mus = [mu1, mu2, mu3]

Sigma1 = np.array([[1.4, 0.0],[0.0, 0.7]])
Sigma2 = np.array([[1.2, 0.0],[0.0, 0.75]])
Sigma3 = np.array([[0.8, 0.0],[0.0, 1.2]])
Sigmas = [Sigma1,Sigma2,Sigma3]

A = create_GMM_adjacency(mus,Sigmas,n,pr) # creatting the adjacency matrix of the graph


# initilization 
S = np.ones(n)  # indicator for susceptible population
I = np.zeros(n) # indicator for infected population
R = np.zeros(n) # indicator for recovered population


sum_S  = np.zeros(nt) # number of susceptible people 
sum_I  = np.zeros(nt)
sum_R  = np.zeros(nt)
sum_dI = np.zeros(nt)

tt = np.arange(nt)

if random_init_infection:
    I  = np.array(np.random.rand(n)<I0,int)
else:
    I[int(n/2):int(n/2)+10] = 1.0

S = S - I 


sum_S[0] = np.sum(S)
sum_I[0] = np.sum(I)
sum_R[0] = np.sum(R)

for t in range(1,nt):
    if t%10==0:
        print(t)
    AI = np.dot(A,I)
    dI = np.array(np.random.rand(n)<S*(1-(1-beta)**(AI)),int)
    dR = np.array(np.random.rand(n)<alpha*I,int)
    dS = np.array(np.random.rand(n)<eta*R,int)
    I  = I + dI - dR
    S  = S - dI + dS
    R  = R + dR - dS

    sum_S[t] = np.sum(S)
    sum_I[t] = np.sum(I)
    sum_R[t] = np.sum(R)
    sum_dI[t] = np.sum(dI) 


# saving the data
data_dict = {'I':sum_I, 'dI':sum_dI, 'S':sum_S, 'R':sum_R}

pickle_f = open('data/GMM_n_%i_m_%i_randm_%.3f_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.pkl' %(n,m,randm,beta,alpha,eta,random_init_infection),"wb")
pickle.dump(data_dict,pickle_f)
pickle_f.close()



 # fitting the model dI = c I^\gamma S^\kappa to data for unknowns c \gamma and \kappa
log_S  = np.log(sum_S) 
log_I  = np.log(sum_I) 
log_dI = np.log(sum_dI)
idx = np.isfinite(log_I)*np.isfinite(log_dI) # the index of non-zero elements 

X = np.vstack((log_I[idx],log_S[idx])).T
Y = log_dI[idx]

reg = LinearRegression().fit(X,Y) # fitting the model 
print (reg.coef_)
gamma = reg.coef_[0]
kappa = reg.coef_[1]
beta_cf = reg.intercept_


# plotting simulation result
plt.figure()

plt.subplot(2,1,1)
plt.fill_between(tt,sum_I,sum_I+sum_S, lw = 2, label = 'S(t)')
plt.fill_between(tt,np.zeros(nt),sum_I, lw = 2, label = 'I(t)')
plt.fill_between(tt,sum_I+sum_S,sum_I+sum_S+sum_R, lw = 2, label = 'R(t)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3, fontsize=15)

plt.subplot(2,1,2)
plt.loglog( sum_I, sum_dI, ls='none', marker='o', ms=4, color='C0', label='simulation data', alpha=0.7 )
plt.loglog( sum_I[idx], np.exp(beta_cf)*(sum_I[idx]**gamma)*(sum_S[idx]**kappa),label=r'$I(t)^{%.2f}S(t)^{%.2f}$' %(gamma,kappa), ls='--', lw=2,color='C1')
plt.legend(fontsize=15)
plt.grid()
plt.xlabel(r'$I$',fontsize=15)
plt.ylabel(r'$\Delta I$',fontsize=15,rotation=0)

plt.savefig('figs/GMM_n_%i_m_%i_randm_%.3f_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.png' %(n,m,randm,beta,alpha,eta,random_init_infection))

plt.figure()
ax = plt.subplot(111,projection='3d')
plt.plot( log_S, log_I, log_dI, ls='none', marker='o', ms=4, color='C0', label='simulation data', alpha=0.7 )
plt.plot( log_S[idx], log_I[idx], beta_cf + gamma*log_I[idx]+ kappa*log_S[idx],label=r'$I(t)^{%.2f}S(t)^{%.2f}$' %(gamma,kappa), ls='--', lw=2,color='C1')
plt.legend(fontsize=15, loc = 2, ncol=2, bbox_to_anchor=(0.01, 1.1))

ax.set_ylabel(r'$\log(I)$',fontsize=15)
ax.set_xlabel(r'$\log(S)$',fontsize=15,rotation=0)
ax.set_zlabel(r'$\log(\Delta I)$',fontsize=15,rotation=0)

plt.savefig('figs/GMM_n_%i_m_%i_randm_%.3f_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i_3d.png' %(n,m,randm,beta,alpha,eta,random_init_infection))



plt.show()
