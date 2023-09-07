<h1>Noisy Randomized Kaczmarz<h1>

This GitHub repository is the official implementation of the numerical experiments presented in the paper _"A Note on Randomized Kaczmarz Algorithm for Solving Doubly-Noisy Linear Systems"_

## Requirements
We conducted the experiments using python 3.8.5, numpy 1.19.2, and scipy 1.8.0 To run the experiments, you can create a Jupyter notebook and simply insert the commands listed in each section.

## Additive noise
To run the experiment use the following command:
```
%run RK_additive_noise.py  --n_iter 500000 --n_run 10 --eta 1
```

To plot the results, use the following code:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sigmas_listA = [0, 0.01, 0.5, 1, 1, 0]
sigmas_listb = [0, 0.01, 0.5, 1, 0, 1]

bound1_results=[]
for e in range(len(sigmas_listA)):
    name ='Results/additive_noise/sigmaA_{}_sigmab_{}'.format(sigmas_listA[e],sigmas_listb[e])
    with open(name+'_boundPartial', "rb") as fp:  bound1_results.append(pickle.load(fp))
means_list1 = []
std_list1 = []
for i in range(len(sigmas_listA)):
    means_list1.append( pd.DataFrame(bound1_results[i]).mean(axis = 0) )
    std_list1.append( pd.DataFrame(bound1_results[i]).std(axis = 0) )

bound2_results=[]
for e in range(len(sigmas_listA)):
    name ='Results/additive_noise/sigmaA_{}_sigmab_{}'.format(sigmas_listA[e],sigmas_listb[e])
    with open(name+'_boundGeneral', "rb") as fp:  bound2_results.append(pickle.load(fp))
means_list2 = []
std_list2 = []
for i in range(len(sigmas_listA)):
    means_list2.append( pd.DataFrame(bound2_results[i]).mean(axis = 0) )
    std_list2.append( pd.DataFrame(bound2_results[i]).std(axis = 0) )

distance_results=[]
for e in range(len(sigmas_listA)):
    name ='Results/additive_noise/sigmaA_{}_sigmab_{}'.format(sigmas_listA[e],sigmas_listb[e])
    with open(name+'_distance_to_x_ls', "rb") as fp:  distance_results.append(pickle.load(fp))
distance_means_list = []
distance_std_list = []
for i in range(len(sigmas_listA)):
    distance_means_list.append( pd.DataFrame(distance_results[i]).mean(axis = 0) )
    distance_std_list.append( pd.DataFrame(distance_results[i]).std(axis = 0) )

colors = ["#672189" for i in range(len(sigmas_listA))]
colorsb = ["black" for i in range(len(sigmas_listA))]
colorsc = ["red" for i in range(len(sigmas_listA))]

fig, ax = plt.subplots(2, 3,  figsize=(24.5, 11.0))
#fig.tight_layout(pad=8.0)

fig.subplots_adjust(hspace=.5)

i = 1

for e in range(len(sigmas_listA)):

    plt.subplot(2, 3, i)

    # plotting the distance
    array_mean = np.array(distance_means_list[e].iloc[:,])
    array_std = np.array(distance_std_list[e].iloc[:,])
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.plot(array_mean, label='Approximation error', color=colors[e], linewidth=2)
    else:
        plt.plot(array_mean, label='Approximation error', color=colors[e], linewidth=2)
    plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2, color=colors[e])


    #plotting the bound of theorem 3.1
    array_mean = np.array(means_list2[e].iloc[:,])
    array_std = np.array(std_list2[e].iloc[:,])
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.plot(array_mean, label="Bound Theorem 3.1", linestyle='dashed', color=colorsb[e], linewidth=2)
    else:
        plt.plot(array_mean, label="Bound Theorem 3.1", linestyle='dashed', color=colorsb[e], linewidth=2)
    plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2, color=colors[e])


    #plotting the bound of theorem 2.5
    array_mean = np.array(means_list1[e].iloc[:,])
    array_std = np.array(std_list1[e].iloc[:,])
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.plot(array_mean, label="Bound Theorem 2.5 ", linestyle='dashdot', color=colorsc[e], linewidth=2)
    else:
        plt.plot(array_mean, label="Bound Theorem 2.5 ", linestyle='dashdot', color=colorsc[e], linewidth=2)
    plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2, color=colors[e])

    
    plt.xlabel("Iterations",fontsize = 27)
    plt.rcParams['xtick.labelsize']=24
    plt.rcParams['ytick.labelsize']=24
    
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.title(r'$\sigma_A=\sigma_b=%s$'%(str(sigmas_listA[e])), fontsize=28)
    else:
        plt.title(r'$\sigma_A=%s,\sigma_b=%s$'%(str(sigmas_listA[e]),str(sigmas_listb[e])), fontsize=28)

    plt.yscale('log')
    plt.grid(linestyle = '--')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))

    if i==1:
        plt.xlim(0,300000)
    elif i==2:
        plt.xlim(0,500000)
    elif i==3:
        plt.xlim(0,300000)
    elif i==4:
        plt.xlim(0,300000)
    elif i==5:
        plt.xlim(0,500000)
    else:
        plt.xlim(0,300000)

    i += 1

plt.legend(bbox_to_anchor=(-2.4, 2.71, 3.4, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, fontsize=28)

```

## Multiplicative noise
To run the experiments use the command below, use zeroF=True (respectively, zeroE=True) to set the noise E (respectively, F) to zero.
```
%run RK_multiplicative_noise.py  --n_iter 300000 --n_run 10 --eta 1 --zeroF=True --zeroE=False
```

To plot the results, use the following code:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sigmas_listA = [0.005,0.01,0.1]    # change accordingly
sigmas_listb = [0.005,0.01,0.1]    # change accordingly

bounds_results=[]
for e in range(len(sigmas_listA)):
    name ='Results/multiplicative_noise/GenearlCase/sigmaA_{}_sigmab_{}'.format(sigmas_listA[e],sigmas_listb[e])  # change name accordingly
    with open(name+'_bound', "rb") as fp:  bounds_results.append(pickle.load(fp))
means_list = []
std_list = []
for i in range(len(sigmas_listA)):
    means_list.append( pd.DataFrame(bounds_results[i]).mean(axis = 0) )
    std_list.append( pd.DataFrame(bounds_results[i]).std(axis = 0) )

distance_results=[]
for e in range(len(sigmas_listA)):
    name ='Results/multiplicative_noise/GenearlCase/sigmaA_{}_sigmab_{}'.format(sigmas_listA[e],sigmas_listb[e])   # change name accordingly
    with open(name+'_distance_to_x_ls', "rb") as fp:  distance_results.append(pickle.load(fp))
distance_means_list = []
distance_std_list = []
for i in range(len(sigmas_listA)):
    distance_means_list.append( pd.DataFrame(distance_results[i]).mean(axis = 0) )
    distance_std_list.append( pd.DataFrame(distance_results[i]).std(axis = 0) )

colors = ["#672189" for i in range(len(sigmas_listA))]
colorsb = ["black" for i in range(len(sigmas_listA))]

fig, ax = plt.subplots(1, 3,  figsize=(26.5, 5.0))
i = 1
for e in range(len(sigmas_listA)):
    plt.subplot(1, 3, i)
    # plotting the distance
    array_mean = np.array(distance_means_list[e].iloc[:,])
    array_std = np.array(distance_std_list[e].iloc[:,])
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.plot(array_mean, label='Approximation error', color=colors[e], linewidth=2)
    else:
        plt.plot(array_mean, label='Approximation error', color=colors[e], linewidth=2)
    plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2, color=colors[e])
    #plotting the bounds
    array_mean = np.array(means_list[e].iloc[:,])
    array_std = np.array(std_list[e].iloc[:,])
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.plot(array_mean, label="Bound Corollary 3.6", linestyle='dashed', color=colorsb[e], linewidth=2)
    else:
        plt.plot(array_mean, label="Bound Corollary 3.6", linestyle='dashed', color=colorsb[e], linewidth=2)
    plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2, color=colors[e])
       
    plt.xlabel("Iterations",fontsize = 27)
    plt.rcParams['xtick.labelsize']=24
    plt.rcParams['ytick.labelsize']=24
    
    if sigmas_listA[e]==sigmas_listb[e]:
        plt.title(r'$\sigma_A=\sigma_b=%s$'%(str(sigmas_listA[e])), fontsize=28)
    else:
        plt.title(r'$\sigma_A=%s,\sigma_b=%s$'%(str(sigmas_listA[e]),str(sigmas_listb[e])), fontsize=28)

    plt.yscale('log')
    plt.grid(linestyle = '--')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))

    plt.xlim(0,150000)

    i += 1

plt.legend(bbox_to_anchor=(-1.87, 1.15, 2.3, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2, fontsize=28)

```
## Additive Preconditioner

To reproduce the experiments use the following command:

```
%run RK_additive_preconditioner.py
```
