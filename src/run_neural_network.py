import mnist_loader
import digit_classifier_neural_network


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    neural_network = digit_classifier_neural_network.Network([784, 100, 10])
    neural_network.SGD(training_data, 30, 10, 3.0, test_data=test_data)