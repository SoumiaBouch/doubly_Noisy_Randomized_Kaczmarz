import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import linalg 


# generating matrix A & vector b

m = 100  # numbre of rows of A
n = 50   # numbre of columns of A
r = 50   # rank of A

singular_values_A = np.array([1] + [4 for i in range(r-1)])

U = linalg.orth(np.random.randn(m,r))

V = linalg.orth(np.random.randn(n,r))

D = np.diag(singular_values_A)

A = np.linalg.multi_dot([U, D, np.transpose(V)])

var = 1
mu = 4
x_ls = var * np.random.randn(n,) + mu
b = np.dot(A,x_ls)

print("rank(A)  ",np.linalg.matrix_rank(A))
aug = np.column_stack((A,b))
print("rank(A|b)  ",np.linalg.matrix_rank(aug))
print("system Ax=b is consistent:  ", np.linalg.matrix_rank(A)==np.linalg.matrix_rank(aug))

# SVD of A
U, s, Vt = np.linalg.svd(A)
Ur = U[:,r-1].reshape((m,1))
VrT = Vt[r-1,:].reshape((1,n))

# constructing the noise
E = (s[-2]-s[-1])* Ur @ VrT

# noisy matrix
Atld = A + E

# computing R & Rtld
Rtld = (np.linalg.norm(np.linalg.pinv(Atld), ord=2)*np.linalg.norm(Atld, 'fro'))**2
R = (np.linalg.norm(np.linalg.pinv(A), ord=2)*np.linalg.norm(A, 'fro'))**2
print('R: ',R)
print('Rtld: ', Rtld)

# starting point
x_0 = (10**2) * np.random.randn(n)

# defining a function to run the RK algorithm
def run_RK(A, E, b, eta, n_run, n_iter):

    Atld = A + E

    print("cond(A):  ",np.linalg.cond(A))
    print("cond(A_tld):  ",np.linalg.cond(Atld))

    # generating probabilities of choosing the rows
    probas = []
    frob_norm_Atld = np.linalg.norm(Atld, ord='fro')
    for i in range(Atld.shape[0]):
        probas.append((np.linalg.norm(Atld[i], ord=2)**2)/(frob_norm_Atld**2))

    
    # lists to store results 
    distance_to_x_ls = [[] for i in range(int(n_run))]
    bound = [[] for i in range(int(n_run))]
    
    
    for r in range(int(n_run)):
        
        x = x_0
        
        distance_to_x_ls[r].append(np.linalg.norm(x - x_ls)**2)
       
        for i in range(int(n_iter)):
            #row_idx = np.random.randint(0,m)
            row_idx = int(np.random.choice(m, 1, p=probas))
            if np.linalg.norm(Atld[row_idx,:])==0:
                continue
            else:
                x = x + eta*(b[row_idx] - np.dot(Atld[row_idx,:],x))/((np.linalg.norm(Atld[row_idx,:], ord=2))**2)*Atld[row_idx,:]
                distance_to_x_ls[r].append(np.linalg.norm(x - x_ls)**2)
                
        print("end run", r)
    return(distance_to_x_ls)

# runing the RK algorithm
n_run = 10
n_iter = 1500
eta = 1

distance_to_x_ls_noisy = run_RK(A, E, b, eta, n_run, n_iter)
distance_to_x_ls_noise_free = run_RK(A, np.zeros((m,n)), b, eta, n_run, n_iter)

### plotting the results
plt.figure(figsize=(11.0, 6.5))

# plotting the distance
mean_noisy  = pd.DataFrame(distance_to_x_ls_noisy).mean(axis = 0)
std_noisy = pd.DataFrame(distance_to_x_ls_noisy).std(axis = 0)

array_mean = np.array(mean_noisy.iloc[:,])
array_std = np.array(std_noisy.iloc[:,])

plt.plot(array_mean, label="RK on noisy system",  marker='x', markevery=100, linewidth=2)
plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2)

mean_noise_free  = pd.DataFrame(distance_to_x_ls_noise_free).mean(axis = 0)
std_noise_free = pd.DataFrame(distance_to_x_ls_noise_free).std(axis = 0)

array_mean = np.array(mean_noise_free.iloc[:,])
array_std = np.array(std_noise_free[:,])

plt.plot(array_mean, label="RK on noise free system", marker='^', markevery=100, linewidth=2)
plt.fill_between(range(len(array_mean)), array_mean-0.5*array_std, array_mean+0.5*array_std, alpha=0.2)

plt.xlabel("Iterations",fontsize = 27)
plt.ylabel('Approximation error',fontsize = 27)
plt.rcParams['xtick.labelsize']=24
plt.rcParams['ytick.labelsize']=24
#plt.xlim(0,1400)
plt.legend(loc='upper right',fontsize=26)
plt.yscale('log')
plt.grid(linestyle = '--')

plt.show()
