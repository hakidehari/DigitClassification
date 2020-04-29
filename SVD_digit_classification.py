"""
    A digit classification Machine Learning module which utilizes 
    Singular Value Decomposition to predict Digits based on their matrix values

    Author:  Haki Dehari
"""
import numpy as np
import math

class DigitClassification(object):

    def __init__(self):
        '''Initializes training and test set into numpy arrays'''
        self.training_set = self.load_data("zip.train")
        self.test_set = self.load_data("zip.test")


    def load_data(self, filename):
        '''Loads data from the training set into a numpy array'''
        data = np.loadtxt(filename)
        return data

    
    def split_training_sets_by_number(self, training_set):
        '''splits every training item in the training set into 3D arrays for each digit'''
        print("Splitting up the training set by digit...")
        #initialize the array which will group each digit training set
        D = [[] for i in range(0, 10)]
        for i in range(0, len(training_set)):
            new_arr = []
            for j in range(0, len(training_set[i])):
                new_arr.append(training_set[i][j])
            D[int(training_set[i][0])].append(new_arr)
        #convert D to numpy array
        D = np.array(D)
        #transpose elements to form columns out of the rows for each digit in D
        for i in range(0, len(D)):
            D[i] = np.transpose(D[i])
        print("Finished splitting up the training set by digit.")
        return D


    def list_training_set_by_digit(self, D):
        '''lists the length of each digit training set'''
        for i in range (0, len(D)):
            print("Number of training examples for digit {0}:    {1}".format(i, len(D[i][0])))

    
    def compute_svd(self, D):
        '''computes the SVD for each digit training set'''
        print("Computing SVD for each digit matrix...")
        svd_dict = {}
        for i in range(0, len(D)):
            U, S, V = np.linalg.svd(D[i], full_matrices=False)
            svd_dict[i] = (U, np.diag(S), V)
        print("Computing SVD complete.")
        return svd_dict


    def clean_digit_array(self, D):
        '''remove the first row of each digit training set(which contains the digit) after we transpose each 2d array
            in the previous function'''
        print("Cleaning up each 2D digit array...")
        for i in range(0, len(D)):
            D[i] = np.delete(D[i], 0, 0)
        print("Finished cleaning up each 2D digit array.")
        return D


    def compute_ranks(self, D):
        '''Computes ranks for each digit matrix'''
        return [np.linalg.matrix_rank(D[i]) for i in range(0, 10)]

    
    def split_test_set(self, test_data):
        '''Split test data by digit and matrix'''
        y = test_data[:, 0]
        data = test_data[:, 1:]
        return (y, data)

    
    def find_rank_k_approximation(self, svd_dict, D, ranks):
        '''
            Approximates the best rank - K approximation for each digit matrix
            which will then be used during execution of the classification function
            to predict the digit
        '''
        print("Approximating k for each matrix")
        k_dict = {}
        for i in range(0, 10):
            S_shape = svd_dict[i][1].shape
            k = 1
            min_value = 99999999
            while k < ranks[i]:
                S_k = np.concatenate((svd_dict[i][1][:, :k], np.zeros((S_shape[0], S_shape[1]-k))), 1)
                U_i = svd_dict[i][0]
                V_i = svd_dict[i][2]
                D_k = np.matmul(np.matmul(U_i, S_k), V_i)
                matrix_difference = np.subtract(D[i], D_k)
                total = np.linalg.norm(matrix_difference, 'fro')
                if total < min_value:
                    min_value = total
                    k_dict[i] = k
                k += 1
        print("Finished approximating k.")
        print(k_dict)
        return k_dict



    def run_classification(self, svd_dict, test_data, y, k_dict):
        '''
            Uses the rank - K approximation of each 2d digit matrix to predict
            the digit based on U_k approximation for each digit U in the singular 
            value decomposition
        '''
        print("Beginning classification")
        count_classified = 0
        count = 0
        for j in range(0, len(test_data)):
            min_value = 9999999
            min_i = 0
            identity = np.identity(256)
            for i in range(0, 10):
                U_i = svd_dict[i][0][:, :10]
                inner = np.subtract(identity, U_i.dot(U_i.T))
                value_to_norm = np.matmul(inner, test_data[j])
                final_value = pow(np.linalg.norm(value_to_norm), 2)
                if final_value < min_value:
                    min_value = final_value
                    min_i = i
            count_classified += 1
            #print("Classified {0} out of {1} test cases".format(count_classified, len(test_data)))
            if int(y[j]) == min_i:
                count += 1
        print("Classified {0}/{1} test sets correctly".format(count, len(test_data)))


if __name__ == '__main__':
    #read in training data
    dc = DigitClassification()

    print("Printing training set...")
    print(dc.training_set)

    #split training set by digit.  Access by using the digit as the index for the array.
    training_set_split_by_digit = dc.split_training_sets_by_number(dc.training_set)

    #list number of training images for each digit
    dc.list_training_set_by_digit(training_set_split_by_digit)

    #clean first rows for digit matrices
    training_set_split_by_digit = dc.clean_digit_array(training_set_split_by_digit)

    #compute svd for each digit matrix
    svd_dict = dc.compute_svd(training_set_split_by_digit)
    #print(svd_dict[0])
    
    #compute ranks for each digit 2D array
    ranks = dc.compute_ranks(training_set_split_by_digit)

    #compute rank k approximation for each digit
    k_dict = dc.find_rank_k_approximation(svd_dict, training_set_split_by_digit, ranks)

    #load test data
    print("Printing test set...")
    print(dc.test_set)

    #split test data
    y, data = dc.split_test_set(dc.test_set)

    #run classification
    dc.run_classification(svd_dict, data, y, k_dict)
    


    