import numpy as np
import math

class DigitClassification:


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
        return D


    def list_training_set_by_digit(self, D):
        '''lists the length of each digit training set'''
        for i in range (0, len(D)):
            print("Number of training examples for digit {0}:    {1}".format(i, len(D[i][0])))

    
    def compute_svd(self, D):
        '''computes the SVD for each digit training set'''
        svd_dict = {}
        for i in range(0, len(D)):
            U, S, V = np.linalg.svd(D[i], full_matrices=True)
            svd_dict[i] = (U, np.diag(S), V)
        return svd_dict


    def clean_digit_array(self, D):
        '''remove the first row of each digit training set(which contains the digit) after we transpose each 2d array
            in the previous function'''
        for i in range(0, len(D)):
            D[i] = np.delete(D[i], 0, 0)
        return D


    def compute_rank_by_digit(self, D):
        '''Computes ranks for each digit 2D array'''
        return [np.linalg.matrix_rank(matrix) for matrix in D]

    
    def compute_rank_k_approximations(self, D, ranks, svd_dict):
        '''
            This function will select a k approximation which is minimum value of the frobenius norm
            given a k value < r.
            Of course, the closer to r that k is the better the approximation will be
            This is used to save time and space when computing SVD's
        '''
        #initialize return dict
        min_k = {}
        for i in range(0, len(D)):
            print("Computing rank-k approximation for digit: {0}".format(i))
            minimum = 99999999
            k = 2
            while k < ranks[i]:
                U_k = svd_dict[i][0][:, 1:k]
                S_k = svd_dict[i][1][:, 1:k]
                V_k = svd_dict[i][2][:, 1:k]
                US_k = U_k * S_k
                D_k = np.dot(US_k, np.transpose(V_k))
                matrix_difference = D[i] - D_k
                total = 0
                for j in range(0, len(matrix_difference)):
                    for l in range(0, len(matrix_difference[j])):
                        total += pow(matrix_difference[j][l], 2)
                total = math.sqrt(total)
                if total < minimum:
                    minimum = total
                    min_k[i] = k
                k += 1
            print("The best rank-k approximation and value of k for digit {0}: {1}".format(i, min_k[i]))
        return min_k



        

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
    ranks = dc.compute_rank_by_digit(training_set_split_by_digit)

    #compute rank-k approximations
    rank_k_approximations = dc.compute_rank_k_approximations(training_set_split_by_digit, ranks, svd_dict)

    #load test data
    print("Printing test set...")
    print(dc.test_set)
    

    