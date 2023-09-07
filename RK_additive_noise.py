# importing necessary libraries
import numpy as np
from scipy import linalg
import pickle
import sys
import getopt


# defining a function to run the Randomized Kaczmarz (RK) algorithm
def run_RK(A, E, b, e, n_run, n_iter, eta, sigma_A, sigma_b):

    """
    The arguments for the run_RK() function are the following:
    A            : The non noisy matrix
    E            : The noise added to the matrix A
    b            : The non noisy right-hand side vector
    e            : The noise added to the vector b
    n_run        : numbre of runs of the RK algorithm to be performed
    n_iter       : numbre of iterations in each run
    eta          : The learning rate
    sigma_A      : The left-hand noise magnitude
    sigma_b      : The right-hand noise magnitude

    """

    A_tld = A+sigma_A*E         # The noisy matrix
    b_tld = b+sigma_b*e      # The noisy right-hand side vector


    print("cond(A_tld):  ",np.linalg.cond(A_tld))

    print("norm of noise on the matrix A:  ",np.linalg.norm(sigma_A*E, ord=2))

    print("||pinv(A)||*||sigma_A*E|| < 1:    ", np.linalg.norm(sigma_A*E, ord=2)<(1/np.linalg.norm(np.linalg.pinv(A), ord=2)))

    print("rank(A_tld)  ",np.linalg.matrix_rank(A_tld))
    augg = np.column_stack((A_tld,b))
    print("rank(A_tld|b)  ",np.linalg.matrix_rank(augg))
    print("system A_tld x = b is consistent:  ", np.linalg.matrix_rank(A_tld)==np.linalg.matrix_rank(augg))

    # computing x_pnls
    x_pnls,res,rank,s = linalg.lstsq(A_tld, b)

    # computing R_tld
    R = (np.linalg.norm(np.linalg.pinv(A_tld), ord=2)*np.linalg.norm(A_tld, 'fro'))**2


    # coputing norms
    norm_pinvA = np.linalg.norm(np.linalg.pinv(A), ord=2)
    norm_sigmaE = np.linalg.norm(sigma_A*E, ord=2)
    norm_sigmae = np.linalg.norm(sigma_b*e, ord=2)

    # computing the min singular value of A_tld: lambda_min
    U, s, Vt = np.linalg.svd(A_tld)
    lambda_min = min(s)

    # computing the horizons of the theoretical bounds, horizon1 is for the bound of theorem 2.5, horizon2 for theorem 3.1
    horizon1 = 2*(np.linalg.norm(x_ls, ord=2))* (norm_pinvA*norm_sigmaE) / (1-norm_pinvA*norm_sigmaE) + norm_sigmae/lambda_min
    horizon2 = (np.linalg.norm(sigma_A*E @ x_ls - sigma_b*e , ord=2)/lambda_min)

    # generating probabilities of choosing the rows
    probas = []
    frob_norm_Atld = np.linalg.norm(A_tld, ord='fro')
    for i in range(A_tld.shape[0]):
        probas.append((np.linalg.norm(A_tld[i], ord=2)**2)/(frob_norm_Atld**2))

    
    # lists to store results 
    distance_to_x_ls = [[] for i in range(int(n_run))]
    bound1 = [[] for i in range(int(n_run))]   # Theoretical bound of theorem 2.5
    bound2 = [[] for i in range(int(n_run))]   # Theoretical bound of theorem 3.1
       
    # generate starting point x_0, must be in the column space of Atld^T
    x_0 = np.transpose(A_tld) @ np.random.normal(size=(m,))
    print("shape of x_0:  ", x_0.shape)

    StartingPoints.append(x_0)
 
    # runing the RK algorithm
    for r in range(int(n_run)):

        # starting point
        x = x_0
        
        distance_to_x_ls[r].append(np.linalg.norm(x - x_ls))
        if r==0:
            bound1[r].append(np.linalg.norm(x_0 - x_pnls, ord=2) + horizon1)
            bound2[r].append(np.linalg.norm(x_0 - x_ls, ord=2)+ horizon2)


        for i in range(int(n_iter)):
            #row_idx = np.random.randint(0,m)
            row_idx = int(np.random.choice(m, 1, p=probas))
            if np.linalg.norm(A_tld[row_idx,:])==0:
                continue
            else:
                x = x + eta*(b_tld[row_idx] - np.dot(A_tld[row_idx,:],x))/((np.linalg.norm(A_tld[row_idx,:], ord=2))**2)*(A_tld[row_idx,:].T)
                distance_to_x_ls[r].append(np.linalg.norm(x - x_ls))
                if r==0:
                    bound1[r].append(((1-1/R)**((i+1)/2))*np.linalg.norm(x_0 - x_pnls, ord=2) + horizon1)
                    bound2[r].append(((1-1/R)**((i+1)/2))*(np.linalg.norm(x_0 - x_ls, ord=2)) + horizon2)
        print("end run", r)
     
    # saving results
    name ='Results/additive_noise/sigmaA_{}_sigmab_{}'.format(sigma_A,sigma_b)
    with open(name+'_distance_to_x_ls', "wb") as fp: pickle.dump(distance_to_x_ls,fp)
    with open(name+'_boundPartial', "wb") as fp: pickle.dump(bound1,fp)
    with open(name+'_boundGeneral', "wb") as fp: pickle.dump(bound2,fp)


def main(argv):

    n_iter = None
    n_run = None
    eta = None

    try:
        opts, args = getopt.getopt(argv[1:], '', ["n_iter=", "n_run=", "eta=" ])
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



    # generating matrix A

    global m,n
    m = 500   # numbre of rows of A
    n = 300   # numbre of columns of A
    r = 300   # rank of A

    sigma_min = 5    # minimum singular value of A
    sigma_max = 50   # maximum singular value of A

    singular_values = np.linspace(sigma_min,sigma_max,r)

    U = linalg.orth(np.random.randn(m,r))

    V = linalg.orth(np.random.randn(n,r))

    D = np.diag(singular_values)

    A = np.linalg.multi_dot([U, D, np.transpose(V)])

    print("cond(A):  ",np.linalg.cond(A))

    # generating vector b
    x_opt = np.random.randn(n,)    
    b = np.dot(A,x_opt)

    # computing x_ls
    global x_ls
    x_ls = np.dot(np.linalg.pinv(A),b)

    # checking consistency of noise free system
    print("rank(A)  ",np.linalg.matrix_rank(A))
    aug = np.column_stack((A,b))
    print("rank(A|b)  ",np.linalg.matrix_rank(aug))
    print("system Ax=b is consistent:  ", np.linalg.matrix_rank(A)==np.linalg.matrix_rank(aug))


    # generating the right-hand noise
    noiseb = np.random.randn(m,)

    # generating the left-hand noise
    x = np.random.randn(n,)
    Atld = np.dot(b.reshape(m,1),np.transpose(x.reshape(n,1)))
    noiseA = Atld-A
    noiseA = (1/np.linalg.norm(noiseA,ord=2))*noiseA
  
    # save generated data and noise
    with open("Results/additive_noise/Matrix_A", "wb") as fp: pickle.dump(A,fp)
    with open("Results/additive_noise/vector_b", "wb") as fp: pickle.dump(b,fp)
    with open("Results/additive_noise/noiseA", "wb") as fp: pickle.dump(noiseA,fp)
    with open("Results/additive_noise/noiseb", "wb") as fp: pickle.dump(noiseb,fp)


    global StartingPoints  # storing starting points if needed for further testing
    StartingPoints = [] 
    
    # runing RK with different noise magnitudes
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 1, 1)
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 0, 0)
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 0.01, 0.01)
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 0.5, 0.5)
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 1, 0)
    run_RK(A, noiseA, b, noiseb, n_run, n_iter, eta, 0, 1)


    with open("Results/additive_noise/StartingPoints", "wb") as fp: pickle.dump(StartingPoints,fp)
 
if __name__ == "__main__":
    main(sys.argv)


