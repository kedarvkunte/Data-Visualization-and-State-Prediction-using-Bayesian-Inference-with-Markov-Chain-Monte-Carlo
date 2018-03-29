# Import python modules
import numpy as np
import kaggle
import scipy.stats as st
import matplotlib.pyplot as plt
from random import choice
from random import shuffle
from random import uniform
import secrets
import random



def J_given_alpha(J,alpha):
    I = 0
    if J[0] == 0:
        I = 1
    else:
        I = 1e-50
    Xi = []
    for index,j in enumerate(J):
        if index<(len(J)-1):
            if J[index] == J[index+1]:
                Xi.append(alpha)
            else:
                Xi.append(1 - alpha)
    Prob_J_given_alpha = I
    for index in range(0,len(J) - 1):
        Prob_J_given_alpha *= Xi[index]
    return Prob_J_given_alpha


def B_given_J(B,J):
    Prob_B_given_J_array = []
    for index,value in enumerate(B):
        ##### Jar 0
        if J[index] == 0:
            # white ball in Jar 0
            if B[index] == 0:
                Prob_B_given_J_array.append(0.2)
            # black ball in Jar 0
            else:
                Prob_B_given_J_array.append(0.8)

        ##### Jar 1
        else:
            # white ball in Jar 1
            if B[index] == 0:
                Prob_B_given_J_array.append(0.9)
            # black ball in Jar 1
            else:
                Prob_B_given_J_array.append(0.1)

    Prob_B_given_J = 1
    for index,value in enumerate(B):
        Prob_B_given_J *= Prob_B_given_J_array[index]


    return Prob_B_given_J




def P_alpha(alpha):
    if alpha>=0 and alpha <=1:
        return 1
    else:
        return 1e-50

def P_alpha_J_B(J,B,alpha):
    #print(" Prob_J_given_alpha = ",J_given_alpha(J,alpha))
    return P_alpha(alpha)*B_given_J(B,J)*J_given_alpha(J,alpha)


def J_flip(J):
    J_new = np.copy(J)
    i = secrets.randbelow(len(J_new))
    if J_new[i] == 0:
        J_new[i] = 1
    else:
        J_new[i] = 0

    return J_new


def MH_J_given_alpha_B(B,alpha,iterations):
    alpha_mean = 1e-50
    J_mean = np.zeros(len(B))

    J = J_mean
    for i,val in enumerate(range(iterations)):
        J_new = J_flip(J)
        acceptance_ratio = P_alpha_J_B(J_new,B,alpha)/P_alpha_J_B(J,B,alpha)
        if np.random.rand() <= acceptance_ratio:
            J = J_new

        J_mean = J_mean + J
    #print("Shape of J_mean",J_mean.shape)

    return J_mean/iterations

def alpha_new_random_uniform(alpha): #proposal function for alpha
    alpha_old = alpha
    alpha_new = np.random.rand()
    #print("alpha_new from random uniform distribution is ", alpha_new)
    return alpha_new

def MH_alpha_given_J_B(J,B,iterations):
    alpha_mean = 1e-50
    alpha = alpha_mean

    J_mean = np.zeros(len(B))

    alpha_array_1h = np.empty(0)
    for i in range(iterations):
        alpha_new = alpha_new_random_uniform(alpha)

        #print(" P_alpha_J_B(J, B, alpha) = ", P_alpha_J_B(J, B, alpha))
        acceptance_ratio = P_alpha_J_B(J, B, alpha_new) / P_alpha_J_B(J, B, alpha)


        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new


        alpha_mean = alpha_mean + alpha

    return alpha_mean/iterations


def Alpha_array_1H(J,B,iterations):
    alpha_mean = 1e-50
    alpha = alpha_mean

    J_mean = np.zeros(len(B))

    alpha_array_1h = np.empty(0)
    for i in range(iterations):
        alpha_new = alpha_new_random_uniform(alpha)

        # print(" P_alpha_J_B(J, B, alpha) = ", P_alpha_J_B(J, B, alpha))
        acceptance_ratio = P_alpha_J_B(J, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new

        alpha_array_1h = np.append(alpha_array_1h,alpha)
        alpha_mean = alpha_mean + alpha

    return alpha_array_1h


def proposal_alpha_J(alpha,J):
    return (alpha_new_random_uniform(alpha),J_flip(J))

def MH_alpha_J_given_B(B,iterations):
    alpha_mean = np.random.rand()
    alpha = alpha_mean

    J_mean = np.zeros(len(B))
    J = J_mean

    for i in range(iterations):
        alpha_new, J_new = proposal_alpha_J(alpha,J)
        acceptance_ratio = P_alpha_J_B(J_new, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new

        alpha_mean = alpha_mean + alpha
        J_mean = J_mean + J

    return (J_mean/iterations,alpha_mean/iterations)

def Alpha_Part_1j(B,iterations):
    alpha_mean = np.random.rand()
    alpha = alpha_mean

    J_mean = np.zeros(len(B))
    J = J_mean

    alpha_arr = np.empty(0)
    for i in range(iterations):
        alpha_new, J_new = proposal_alpha_J(alpha,J)
        acceptance_ratio = P_alpha_J_B(J_new, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new

        alpha_arr = np.append(alpha_arr,alpha)
        alpha_mean = alpha_mean + alpha
        J_mean = J_mean + J


    return alpha_arr



def f(Jn, alpha):
    Jn_plus_1 = [0, 1]

    s = 0
    for j in Jn_plus_1:

        si = 1
        if j == Jn:
            si = si*alpha
        else:
            si = si*(1 - alpha)

        if j == 0:
            si = si*0.8
        else:
            si = si*0.1

        s = s + si

    return s

def P_Bn_plus_1_given_Jn_alpha(B,iterations):

    alpha = np.random.rand()

    J = np.zeros(len(B))

    B_mean = 0

    for i in range(iterations):
        alpha_new, J_new = proposal_alpha_J(alpha, J)
        acceptance_ratio = P_alpha_J_B(J_new, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new

        Jn = J[len(B)-1]
        B_mean = B_mean + f(Jn,alpha)

    return B_mean/iterations


def Array_B_mean(B,iterations):

    b_mean_array = np.empty(0)
    alpha = np.random.rand()

    J = np.zeros(len(B))

    B_mean = 0

    for i in range(iterations):
        alpha_new, J_new = proposal_alpha_J(alpha, J)
        acceptance_ratio = P_alpha_J_B(J_new, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new

        Jn = J[len(B)-1]
        B_mean = B_mean + f(Jn,alpha)
        b_mean_array = np.append(b_mean_array,B_mean)

    return b_mean_array/iterations

def my_shuffle_array(array):
    random.shuffle(array)
    return array

def EC_J_Shuffle(J):
    #J_new = np.copy(J)
    J_new = J.copy()
    for jlen in range(0,len(J),int(len(J)/2)):
        J_new = J_flip(J_new)


    #    print("random.sample(J_new,len(J_new)) = ",random.sample(J_new,len(J_new)))
    new_J = np.random.choice(J_new,len(J_new)) #p =[float(1/len(J_new)) for index in range(len(J_new))]
    return my_shuffle_array(new_J)

def EC_alpha_new():
    # # x = random.randint(1, 100)
    # # alpha_new = 0.5 + np.arctan(x)/np.pi
    # alpha_new = random.gauss(0.5, 0.2)
    # while (0 <= alpha_new <= 1) != True:
    #     alpha_new = random.gauss(0.5, 0.2)

    alpha_new = st.norm.cdf(uniform(-2.5,2.5)) #
    #print("alpha_new from random uniform distribution is ", alpha_new)
    return alpha_new


def EC_proposal_alpha_J(alpha,J):
    return (EC_alpha_new(),EC_J_Shuffle(J))


def EC_P_Bn_plus_1_given_Jn_alpha(B,iterations):

    alpha = np.random.rand()

    J = np.zeros(len(B))

    B_mean = 0

    for i in range(iterations):
        alpha_new, J_new = EC_proposal_alpha_J(alpha, J)
        #print("J_new  = ", J_new, " Alpha_new = ",alpha_new)
        acceptance_ratio = P_alpha_J_B(J_new, B, alpha_new) / P_alpha_J_B(J, B, alpha)

        if np.random.rand() <= acceptance_ratio:
            alpha = alpha_new
            J = J_new

        Jn = J[len(B)-1]
        B_mean = B_mean + f(Jn,alpha)

    return B_mean/iterations






################################################
if __name__ == "__main__":

    print('1a through 1l computation goes here ...')

    print('\nProblem 1a \n')
    J = [0, 1, 1, 0, 1]
    alpha = 0.75
    print("Prob_J_given_alpha = ", J_given_alpha(J, alpha))

    J = [0, 0, 1, 0, 1]
    alpha = 0.2
    print("Prob_J_given_alpha = ", J_given_alpha(J, alpha))


    ########### Case 3 ################
    print("\ncase 3:\n")

    J = [1, 1, 0, 1, 0, 1]
    alpha = 0.2
    print("Prob_J_given_alpha = ", J_given_alpha(J, alpha))

    ########### Case 4 ################
    print("\ncase 4:\n")

    J = [0, 1, 0, 1, 0, 0]
    alpha = 0.2
    print("Prob_J_given_alpha = ", J_given_alpha(J, alpha))

    print('\nProblem 1b case 1 \n')
    J = [0, 1, 1, 0, 1]
    B = [1, 0, 0, 1, 1]
    print("Prob_B_given_J = ",B_given_J(B,J))

    print('\nProblem 1b case 2 \n')
    J = [0, 1, 0, 0, 1]
    B = [0, 0, 1, 0, 1]
    print("Prob_B_given_J = ", B_given_J(B, J))

    print('\nProblem 1b case 3 \n')
    J = [0, 1, 1, 0, 0, 1]
    B = [1, 0, 1, 1, 1, 0]
    print("Prob_B_given_J = ", B_given_J(B, J))

    print('\nProblem 1b case 4 \n')
    J = [1, 1, 0, 0, 1, 1]
    B = [0, 1, 1, 0, 1, 1]
    print("Prob_B_given_J = ", B_given_J(B, J))


    print("\nProblem 1d")

    print('\nProblem 1d case 1 \n')
    J = [0, 1, 1, 0, 1]
    B = [1, 0, 0, 1, 1]
    alpha = 0.75
    print("Prob_alpha_J_B = ",P_alpha_J_B(J,B,alpha))

    print('\nProblem 1d case 2 \n')
    J = [0, 1, 0, 0, 1]
    B = [0, 0, 1, 0, 1]
    alpha = 0.3
    print("Prob_alpha_J_B = ", P_alpha_J_B(J, B, alpha))

    print('\nProblem 1d case 3 \n')
    J = [0, 0, 0, 0, 0, 1]
    B = [0, 1, 1, 1, 0, 1]
    alpha = 0.63
    print("Prob_alpha_J_B = ", P_alpha_J_B(J, B, alpha))

    print('\nProblem 1d case 4 \n')
    J = [0, 0, 1, 0, 0, 1, 1]
    B = [1, 1, 0, 0, 1, 1, 1]
    alpha = 0.23
    print("Prob_alpha_J_B = ", P_alpha_J_B(J, B, alpha))

    print('\nProblem 1e \n')

    J = [0, 1, 1, 0, 1]
    print("After flipping J_new = ",J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))
    print("After flipping J_new = ", J_flip(J))

    print('\nProblem 1f Case 1 \n')
    B = [1, 0, 0, 1, 1]
    alpha = 0.5
    iterations = 10000
    result = MH_J_given_alpha_B(B,alpha,iterations)
    print("MH algorithm Prob_J_given_alpha_B = ",result)
    num0 = 1 - result[1]
    num1 = result[1]
    J_val = ["J2=0|α,B","J2=1|α,B"]
    plt.bar([0,1], [num0, num1],align='center',color="blue")
    plt.ylabel('P(J2|alpha,B)')
    plt.xlabel('J2')
    plt.xticks([0,1],J_val)
    plt.title('Bar chart 1f')

    plt.show()

    print('\nProblem 1f Case 2 \n')
    B = [1, 0, 0, 0, 1, 0, 1, 1]
    alpha = 0.11
    iterations = 10000
    print("MH algorithm Prob_J_given_alpha_B = ", MH_J_given_alpha_B(B, alpha, iterations))

    print('\nProblem 1f Case 3 \n')
    B = [1, 0, 0, 1, 1, 0, 0]
    alpha = 0.75
    iterations = 10000
    print("MH algorithm Prob_J_given_alpha_B = ", MH_J_given_alpha_B(B, alpha, iterations))




    alpha_new_random_uniform(alpha)
    alpha_new_random_uniform(alpha)

    print('\nProblem 1h ')
    print('Case 1 \n')
    J = [0, 1, 0, 1, 0]
    B = [1, 0, 1, 0, 1]
    iterations = 10000
    print("MH algorithm Prob_alpha_given_J_B = ", MH_alpha_given_J_B(J,B,iterations))

    print('\nProblem 1h ')
    print('Case 2 \n')
    J = [0, 0, 0, 0, 0]
    B = [1, 1, 1, 1, 1]
    iterations = 10000
    print("MH algorithm Prob_alpha_given_J_B = ", MH_alpha_given_J_B(J, B, iterations))

    print('\nProblem 1h ')
    print('Case 3 \n')
    J = [0, 1, 1, 0, 1]
    B = [1, 0, 0, 1, 1]
    iterations = 10000
    print("MH algorithm Prob_alpha_given_J_B = ", MH_alpha_given_J_B(J, B, iterations))

    plt.hist(Alpha_array_1H(J,B,iterations),bins=50)
    plt.xlabel('alpha')
    plt.ylabel('P(alpha given J and B)')
    plt.title('Histogram part 1h')
    plt.show()




    print('\nProblem 1h ')
    print('Case 4 \n')
    J = [0, 1, 1, 1, 1, 1, 1, 0]
    B = [1, 0, 0, 1, 1, 0, 0, 1]
    iterations = 10000
    print("MH algorithm Prob_alpha_given_J_B = ", MH_alpha_given_J_B(J, B, iterations))

    print('\nProblem 1h ')
    print('Case 5 \n')
    J = [0, 1, 1, 0, 1, 0]
    B = [1, 0, 0, 1, 1, 1]
    iterations = 10000
    print("MH algorithm Prob_alpha_given_J_B = ", MH_alpha_given_J_B(J, B, iterations))

    print('\nProblem 1j ')

    B = [1, 1, 0, 1, 1, 0, 0, 0]
    iterations = 10000
    J,alpha = MH_alpha_J_given_B(B,iterations)
    y_pos = np.arange(len(B))
    plt.bar(y_pos,J,align='center')
    plt.xticks(y_pos,B)

    plt.xlabel('B')
    plt.ylabel('P(J|B)')
    plt.title('Bar chart Part 1J')
    plt.show()

    plt.hist(Alpha_Part_1j(B,iterations), bins=50)
    plt.xlabel('alpha_given_B')
    plt.ylabel('Occurences')
    plt.title('Histogram part 1J')
    plt.show()

    plt.plot(range(iterations), Alpha_Part_1j(B,iterations))
    plt.xlabel('No. of Iterations')
    plt.ylabel('Alpha')
    plt.title('Plot of alpha as a function of iterations Part 1J')
    plt.xlim(-10, 10000)  # set x axis range
    plt.ylim(0, 1)  # Set yaxis range

    plt.show()




    alpha_partj_arr = np.empty(0)
    for index in range(1,iterations+1,500):
        J_partj, alpha_partj = MH_alpha_J_given_B(B, index)
        alpha_partj_arr = np.append(alpha_partj_arr,alpha_partj)

    plt.plot(range(1,iterations+1,500),alpha_partj_arr )
    plt.xlabel('No. of Iterations')
    plt.ylabel('Alpha')
    plt.title('Plot of alpha as a function of iterations Part 1J')
    plt.xlim(-10,10000)  # set x axis range
    plt.ylim(0, 1)  # Set yaxis range

    plt.show()



    print('\nProblem 1k \n')
    print('\nCase 1 \n')
    Jn=1
    alpha=0.6
    print("f(Jn,alpha)for case 1",f(Jn,alpha))

    print('\nProblem 1k \n')
    print('\nCase 2 \n')
    Jn = 0
    alpha = 0.99
    print("f(Jn,alpha)for case 2 %.6f"% f(Jn, alpha))

    print('\nProblem 1k \n')
    print('\nCase 3 \n')
    Jn = 0
    alpha = 0.33456
    print("f(Jn,alpha)for case 3 %.6f"% f(Jn, alpha))

    print('\nProblem 1k \n')
    print('\nCase 4 \n')
    Jn = 1
    alpha = 0.5019
    print("f(Jn,alpha)for case 4 %.6f"% f(Jn, alpha))


    print('\nProblem 1l \n')
    print('\nCase 1 \n')
    B = [0, 0, 1]
    iterations = 10000
    print("P_Bn_plus_1_given_Jn_alpha(B,iterations) = ",P_Bn_plus_1_given_Jn_alpha(B,iterations))

    print('\nProblem 1l \n')
    print('\nCase 2 \n')
    B = [0, 1, 0, 1, 0, 1]
    iterations = 10000
    print("P_Bn_plus_1_given_Jn_alpha(B,iterations) = ", P_Bn_plus_1_given_Jn_alpha(B, iterations))

    print('\nProblem 1l \n')
    print('\nCase 3 \n')
    B = [0, 1, 0, 0, 0, 0, 0]
    iterations = 10000
    print("P_Bn_plus_1_given_Jn_alpha(B,iterations) = ", P_Bn_plus_1_given_Jn_alpha(B, iterations))

    print('\nProblem 1l \n')
    print('\nCase 4 \n')
    B = [1, 1, 1, 1, 1]
    iterations = 10000
    print("P_Bn_plus_1_given_Jn_alpha(B,iterations) = ", P_Bn_plus_1_given_Jn_alpha(B, iterations))

    ###############################################
    B = [0, 0, 1, 1, 0, 0, 0, 1, 0, 1] #corresponding to b = 0 and l = 10
    EC_Plot_array = np.empty(0)
    iterations = 10000
    for index in range(1, iterations + 1, 500):
        EC_Plot_array = np.append(EC_Plot_array, EC_P_Bn_plus_1_given_Jn_alpha(B, index))
        print("Iteration = ",index)

    plt.plot(range(1, iterations + 1, 500), EC_Plot_array, 'sb-', linewidth=2)
    plt.xlabel('No. of Iterations')
    plt.ylabel('B_mean')
    plt.title('Extra Credit Plot of B_mean as a function of iterations')
    # plt.legend(labels, loc="best")
    plt.show()



    Plot_array = np.empty(0)

    print('\n\n1m')
    lengths = [10, 15, 20, 25]
    prediction_prob = list()
    array = np.empty(0)
    iterations = 1000000
    for l in lengths:
        B_array = np.loadtxt('../../Data/B_sequences_%s.txt' % (l), delimiter=',', dtype=float)
        for b in np.arange(B_array.shape[0]):
            prediction_prob.append(np.random.rand(1))

            array = np.append(array,P_Bn_plus_1_given_Jn_alpha(B_array[b, :],iterations))
            print('Prob of next entry in ', B_array[b, :], 'is black is', array[-1])
            if b == 0 and l == 10:
                for index in range(1,iterations+1,10000):
                    Plot_array = np.append(Plot_array,P_Bn_plus_1_given_Jn_alpha(B_array[b, :],index))
                    #EC_Plot_array = np.append(EC_Plot_array,EC_P_Bn_plus_1_given_Jn_alpha(B_array[b, :],index))
                    print("Iteration = ",index)

                #labels = ["1m", "Extra Credit"]
                plt.figure(1, figsize=(6, 4))

                plt.plot(range(1,iterations+1,10000), Plot_array,'or-', linewidth=3)
                #plt.plot(range(1,iterations+1,200), EC_Plot_array, 'sb-', linewidth=2)
                plt.xlabel('No. of Iterations')
                plt.ylabel('B_mean')
                plt.title('Plot of B_mean as a function of iterations')
                #plt.legend(labels, loc="best")
                plt.show()






    # Output file location
    file_name = '../Predictions/best.csv'

    # Writing output in Kaggle format
    print('Writing output to ', file_name)
    kaggle.kaggleize(array, file_name)


