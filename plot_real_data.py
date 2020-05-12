import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
import os
from sklearn.linear_model import LinearRegression
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

def fit_linear(x,y):
    XX = np.array([[np.sum(x*x),np.sum(x)],[np.sum(x),1]])
    XY = np.array([np.sum(x*y),np.sum(y)])
    C  = np.dot(np.linalg.inv(XX),XY)
    return C[0],C[1]

parser = argparse.ArgumentParser()

parser.add_argument('--country', type=str, default='Italy', help='country')
parser.add_argument('--state', type=str, default='New York', help='state (use it when contry is US)')

opt = parser.parse_args()
print(opt)


# reading the CSV file
country = opt.country
state   = opt.state
I=[]
with open('data/real_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    for row in csv_reader:
        if row[2]==country:
            if country == 'US':
                if row[1] == state:
                    I.append(float(row[6]))
            else:
                I.append(float(row[6]))


n = len(I) # size of the data
I = np.array(I)
dI = np.zeros(n)
for i in range(1,n):
    dI[i] = I[i] - I[i-1] # the newly ifected people per day

# fitting the model 
log_I  = np.log(I) 
log_dI = np.log(dI)
idx = np.isfinite(log_I)*np.isfinite(log_dI)

gamma , c = fit_linear(log_I[idx],log_dI[idx]) # fitting the model dI = c I^\gamma for c and \gamma

plt.figure()
plt.loglog( I[idx], np.exp(c)*(I[idx]**gamma),label=r'$I(t)^{%.2f}$' %gamma, ls='--', lw=2, color='C1')
plt.loglog( I, dI,ls='none', marker='o', ms=6, color='C0', label='data (%s)' %(country))
plt.legend(fontsize=15)
plt.grid()
plt.xlabel(r'$I$',fontsize=15)
plt.ylabel(r'$\Delta I$',fontsize=15,rotation=0)

plt.savefig('figs/real_data_%s.png' %country)

plt.show()
