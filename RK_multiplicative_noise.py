# importing necessary libraries
import numpy as np 
from scipy import linalg
import pickle 
import sys 
import getopt


# defining a function to run the Randomized Kaczmarz (RK) algorithm
def run_RK(A, E, F, b, e, n_run, n_iter, eta, sigma_A, sigma_b, folder):

    """
    The arguments for the run_RK() function are the following:
    A            : The non noisy matrix
    E            : The multiplicative left-hand noise to the matrix A
    F            : The multiplicative right-hand noise to the matrix A
    b            : The non noisy right-hand side vector  (Ax=b)
    e            : The noise added to the vector b
    n_run        : numbre of runs of the RK algorithm to be performed
    n_iter       : numbre of iterations in each run
    eta          : The learning rate
    sigma_A      : The left-hand noise magnitude
    sigma_b      : The right-hand noise magnitude
    folder       : Foldre to store results

    """

    noiseE = sigma_A*E
    noiseF = sigma_A*F
    noiseb = sigma_b*e
 

    A_tld = np.linalg.multi_dot([np.identity(m)+noiseE, A, np.identity(n)+noiseF])   # The noisy matrix
    b_tld = b+noiseb      # The noisy right-hand side vector

    print("cond(A_tld):  ",np.linalg.cond(A_tld)) 

    # computing R_tld
    R = (np.linalg.norm(np.linalg.pinv(A_tld), ord=2)*np.linalg.norm(A_tld, 'fro'))**2
 
    # compute the min singular value of A_tld: lambda_min
    U, s, Vt = np.linalg.svd(A_tld)
    lambda_min = min(s)

    # computing the horizon of the theoretical bound
    horizon = (np.linalg.norm( (noiseE @ A + A @ noiseF + noiseE @ A @ noiseF) @ x_ls - noiseb, ord=2) / lambda_min)**2
    
    # generating probabilities of choosing the rows
    probas = []
    frob_norm_Atld = np.linalg.norm(A_tld, ord='fro')
    for i in range(A_tld.shape[0]):
        probas.append((np.linalg.norm(A_tld[i], ord=2)**2)/(frob_norm_Atld**2))

    
    # lists to store results 
    distance_to_x_ls = [[] for i in range(int(n_run))]
    bound = [[] for i in range(int(n_run))] 
    
    # generate starting point x_0, must be in the column space of Atld^T
    x_0 = np.transpose(A_tld) @ np.random.normal(size=(m,))

    # runing the RK algorithm
    for r in range(int(n_run)):

        # starting point
        x = x_0
        
        distance_to_x_ls[r].append(np.linalg.norm(x - x_ls)**2)
        if r==0:
            bound[r].append(np.linalg.norm(x_0 - x_ls, ord=2)**2+ horizon)

        for i in range(int(n_iter)):
            #row_idx = np.random.randint(0,m)
            row_idx = int(np.random.choice(m, 1, p=probas))
            if np.linalg.norm(A_tld[row_idx,:])==0:
                continue
            else:
                x = x + eta*(b_tld[row_idx] - np.dot(A_tld[row_idx,:],x))/((np.linalg.norm(A_tld[row_idx,:], ord=2))**2)*A_tld[row_idx,:]
                
                distance_to_x_ls[r].append(np.linalg.norm(x - x_ls)**2)
                if r==0:
                    bound[r].append(((1-1/R)**(i+1))*(np.linalg.norm(x_0 - x_ls, ord=2)**2) + horizon)
            
        print("end run", r) 
 
    # saving results
    name = folder + 'sigmaA_{}_sigmab_{}'.format(sigma_A,sigma_b)
    with open(name+'_distance_to_x_ls', "wb") as fp: pickle.dump(distance_to_x_ls,fp)
    with open(name+'_bound', "wb") as fp: pickle.dump(bound,fp) 
    

def main(argv):

    n_iter = None
    n_run = None
    eta = None
    zeroF = None
    zeroE = None

    try:
        opts, args = getopt.getopt(argv[1:], '', ["n_iter=", "n_run=", "eta=", "zeroF=", "zeroE=" ])
    except:
        print("Error")

    for opt, arg in opts:
        if opt in ['--n_iter']:
            n_iter = arg
        elif opt in ['--n_run']:
            n_run = arg
        elif opt in ['--eta']:
            eta = arg
            eta = float(eta)
        elif opt in ['--zeroF']:
            zeroF = arg
        elif opt in ['--zeroE']:
            zeroE = arg



    # generating matrix A
    global m,n
    m = 500     # numbre of rows of A
    n = 300     # numbre of columns of A
    r = 300     # rank of A

    sigma_min = 1     # minimum singular value of A
    sigma_max = 10    # maximum singular value of A

    singular_values = np.linspace(sigma_min,sigma_max,r)

    U = linalg.orth(np.random.randn(m,r))

    V = linalg.orth(np.random.randn(n,r))

    D = np.diag(singular_values)

    A = np.linalg.multi_dot([U, D, np.transpose(V)])

    print("cond(A):  ",np.linalg.cond(A))

    # generating vector b
    x_opt = np.random.randn(n,)
    b = np.dot(A,x_opt)

    # checking consistency of noise free system
    print("rank(A)  ",np.linalg.matrix_rank(A))
    aug = np.column_stack((A,b))
    print("rank(A|b)  ",np.linalg.matrix_rank(aug))
    print("Noisless linear system is consistent:  ", np.linalg.matrix_rank(A)==np.linalg.matrix_rank(aug))


    # compute x_ls
    global x_ls
    x_ls = np.dot(np.linalg.pinv(A),b)

    # generating the noises
    E = np.random.normal(loc=[0.]*m, scale=[1.]*m, size=(m,m))
    F = np.random.normal(loc=[0.]*n, scale=[1.]*n, size=(n,n))
    e = np.random.randn(m)

    
    # save generated data and noise
    with open("Results/multiplicative_noise/Matrix_A", "wb") as fp: pickle.dump(A,fp)
    with open("Results/multiplicative_noise/vector_b", "wb") as fp: pickle.dump(b,fp)
    with open("Results/multiplicative_noise/noiseE", "wb") as fp: pickle.dump(E,fp)
    with open("Results/multiplicative_noise/noiseF", "wb") as fp: pickle.dump(F,fp)
    with open("Results/multiplicative_noise/e", "wb") as fp: pickle.dump(e,fp)


    # runing RK with different noise magnitudes
    if zeroF == True:     # case F = 0
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta, 1, 1, 'Results/multiplicative_noise/ZeroF/')
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta,  0.5, 0.5, 'Results/multiplicative_noise/ZeroF/')
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta, 0.05, 0.05, 'Results/multiplicative_noise/ZeroF/')
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta,  0.1, 0.1, 'Results/multiplicative_noise/ZeroF/')
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta,  10, 10, 'Results/multiplicative_noise/ZeroF/')
        run_RK(A, E, np.zeros((n,n)), b, e, n_run, n_iter, eta,  0.005, 0.005, 'Results/multiplicative_noise/ZeroF/')

    elif zeroE == True:     # case E = 0
        run_RK(A, np.zeros((m,m)), F, b, e, n_run, n_iter, eta, 0.005, 0.005, 'Results/multiplicative_noise/ZeroE/')
        run_RK(A, np.zeros((m,m)), F, b, e, n_run, n_iter, eta, 0.01, 0.01, 'Results/multiplicative_noise/ZeroE/')
        run_RK(A, np.zeros((m,m)), F, b, e, n_run, n_iter, eta,  0.5, 0.5, 'Results/multiplicative_noise/ZeroE/')

    elif (zeroE == True and zeroF == True):
        print("ERROR, E or F should not be zero")
    else:                 # general case
        run_RK(A, E, F, b, e, n_run, n_iter, eta, 0.005, 0.005, 'Results/multiplicative_noise/GeneralCase/')
        run_RK(A, E, F, b, e, n_run, n_iter, eta,  0.01, 0.01, 'Results/multiplicative_noise/GeneralCase/')
        run_RK(A, E, F, b, e, n_run, n_iter, eta,  0.1, 0.1, 'Results/multiplicative_noise/GeneralCase/')


if __name__ == "__main__":
    main(sys.argv)


