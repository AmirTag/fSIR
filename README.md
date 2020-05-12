# fSIR

Code and dataset for the article [Fractional SIR Epidemiological Models](https://www.medrxiv.org/content/10.1101/2020.04.28.20083865v2)

The paper studies the spread of infection over a network of people according to the SIR model.  
The claim is that the number of infected people grows proportional to fractional power of the current number of infected people. The claim is verified numerically for: 
1. Two dimensional grid
2. Random graph where nodes are samples from a mixture of Gaussian
3. COVID-19 data set provided by John Hopkins University (JHUCSSE) for the period Jan-31-20 to Mar-24-20, for the countries of Italy, Germany, Iran, and France.

## Reproducing the numerical results

#### Figure 2 and 3 (two-dimensional grid with d=1)
```
python two-dim-grid.py --d 1 --beta 0.2 --alpha 0.05 
python two-dim-grid.py --d 1 --beta 0.2 --alpha 0.1 
python two-dim-grid.py --d 1 --beta 0.3 --alpha 0.05 
python two-dim-grid.py --d 1 --beta 0.3 --alpha 0.1 
```

#### Figure 4 and 5 (two-dimensional grid with d=2)
```
python two-dim-grid.py --d 2 --beta 0.2 --alpha 0.05 
python two-dim-grid.py --d 2 --beta 0.2 --alpha 0.1 
python two-dim-grid.py --d 2 --beta 0.3 --alpha 0.05 
python two-dim-grid.py --d 2 --beta 0.3 --alpha 0.1 
```

#### Figure 6 (random graph GMM)
```
python random-graph-GMM.py --m 4 --beta 0.3 --alpha 0.05 
python random-graph-GMM.py --m 4 --beta 0.3 --alpha 0.1
```

#### Figure 7 (dependence of exponent on connectivity)
```
python random-graph-GMM.py --m 4 
python random-graph-GMM.py --m 5
python random-graph-GMM.py --m 6
python random-graph-GMM.py --m 8
python random-graph-GMM.py --m 10
python random-graph-GMM.py --m 4 --randm 0.02
python random-graph-GMM.py --m 4 --randm 0.05
python random-graph-GMM.py --m 4 --randm 0.1
python exponent-connectivity.py 
```

#### Figure 8 (real data)
```
python plot_real_data.py --country Italy
python plot_real_data.py --country Germany
python plot_real_data.py --country France
python plot_real_data.py --country Iran
```
