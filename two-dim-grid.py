'''
    SIR model over two dimensional grid 
    nodes are located on a two-dimensional grid
    each node is connected to the closest neighbors of distance d (d is a parameter)
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
import argparse

def eval_AI(I, d):
    '''
        computing the summation \sum_j A_ij I_j 
        A_ij is non-zero when the node i and j are neighbors
    '''
    (nn,) = np.shape(I)
    AI = np.zeros(nn)

    for i in range(n):
        for j in range(n):
            if i>0:
                AI[i*n+j] += I[(i-1)*n+j]
            if i<n-1:
                AI[i*n+j] += I[(i+1)*n+j] 
            if j>0: 
                AI[i*n+j] += I[i*n+(j-1)] 
            if j<n-1:
                AI[i*n+j] += I[i*n+(j+1)] 

    if d==2:
         for i in range(n):
            for j in range(n):
                if i>1:
                    AI[i*n+j] += I[(i-2)*n+j]
                if i<n-2:
                    AI[i*n+j] += I[(i+2)*n+j] 
                if j>1: 
                    AI[i*n+j] += I[i*n+(j-2)] 
                if j<n-2:
                    AI[i*n+j] += I[i*n+(j+2)]        
    return AI


parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=100, help='size of the nxn grid')
parser.add_argument('--d', type=int, default=1, help='the distance to neighbors ')
parser.add_argument('--nt', type=int, default=50, help='time-steps')
parser.add_argument('--beta', type=float, default=0.3, help='infection rate')
parser.add_argument('--alpha', type=float, default=0.1, help='recovery rate')
parser.add_argument('--I0', type=float, default=0.001, help='initial infectioin probability')
parser.add_argument('--eta', type=float, default=0.01, help='recovery to susceptible rate')
parser.add_argument('--random_init_infection', type=int, default=1, help='0 if the initial infected population is fixed. 1 if the initial infected population is selected randomly')

opt = parser.parse_args()
print(opt)

n  = opt.n  # size of the nxn grid
nn = n*n    # number of total nodes
d  = opt.d    # the distance to neighbors 
nt = opt.nt     # time-steps
beta = opt.beta  # infection rate
alpha = opt.alpha # recovery rate
I0 = opt.I0  # initial infectioin probability
eta = opt.eta  # recovery to susceptible rate

random_init_infection = opt.random_init_infection # 0 if the initial infected population is fixed. 1 if the initial infected population is selected randomly

S = np.ones(nn)  # indicator for susceptible population
I = np.zeros(nn) # indicator for infected population
R = np.zeros(nn) # indicator for recovered population


sum_S  = np.zeros(nt) # number of susceptible people 
sum_I  = np.zeros(nt)
sum_R  = np.zeros(nt)
sum_dI = np.zeros(nt)

tt = np.arange(nt)

# initilzation
if random_init_infection:
    I  = np.array(np.random.rand(nn)<I0,int)
else:
    I[n+int(n/2)] = 1.0

S = S - I 


sum_S[0] = np.sum(S)
sum_I[0] = np.sum(I)
sum_R[0] = np.sum(R)

for t in range(1,nt):
    if t%10==0:
        print(t)
    AI = eval_AI(I , d) 
    dI = np.array(np.random.rand(nn)<S*(1-(1-beta)**(AI)),int)  # identifying the new infected people
    dR = np.array(np.random.rand(nn)<alpha*I,int) # identifying the new recovered people
    dS = np.array(np.random.rand(nn)<eta*R,int)   # identifying the new susceptible people
    I  = I + dI - dR # updating
    S  = S - dI + dS
    R  = R + dR - dS
    sum_S[t] = np.sum(S)
    sum_I[t] = np.sum(I)
    sum_R[t] = np.sum(R)
    sum_dI[t] = np.sum(dI) 

# saving the data
data_dict = {'I':sum_I, 'dI':sum_dI, 'S':sum_S, 'R':sum_R}

pickle_f = open('data/2dgrid_n_%i_d_%i_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.pkl' %(n,d,beta,alpha,eta,random_init_infection),"wb")
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


# plotting the simulation result


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

plt.savefig('figs/2dgrid_n_%i_d_%i_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i.png' %(n,d,beta,alpha,eta,random_init_infection))


plt.figure()
ax = plt.subplot(111,projection='3d')
plt.plot( log_S, log_I, log_dI, ls='none', marker='o', ms=4, color='C0', label='simulation data', alpha=0.7 )
plt.plot( log_S[idx], log_I[idx], beta_cf + gamma*log_I[idx]+ kappa*log_S[idx],label=r'$I(t)^{%.2f}S(t)^{%.2f}$' %(gamma,kappa), ls='--', lw=2,color='C1')
plt.legend(fontsize=15, loc = 2, ncol=2, bbox_to_anchor=(0.01, 1.1))
ax.set_ylabel(r'$\log(I)$',fontsize=15)
ax.set_xlabel(r'$\log(S)$',fontsize=15,rotation=0)
ax.set_zlabel(r'$\log(\Delta I)$',fontsize=15,rotation=0)

plt.savefig('figs/2dgrid_n_%i_d_%i_beta_%.2f_alpha_%.2f_eta_%.2f_random_init_%i_3d.png' %(n,d,beta,alpha,eta,random_init_infection))


plt.show()
 



