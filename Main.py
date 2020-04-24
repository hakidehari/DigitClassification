import numpy as np


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
        '''splits every training item in the training set into 2d arrays for each digit'''
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
            svd_dict[i] = (U, S, V)
        return svd_dict


    def clean_digit_array(self, D):
        '''remove the first row of each digit training set(which contains the digit) after we transpose each 2d array
            in the previous function'''
        for i in range(0, len(D)):
            D[i] = np.delete(D[i], 0, 0)
        return D
        

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
    print(svd_dict[0])
    
    #load test data
    print("Printing test set...")
    print(dc.test_set)
    

    