import ReadMNIST as mnist
import pickle
import numpy as np

data = mnist.read_mnist_data('test')[0:100].reshape(100,28*28).T

output = open('sample_data.pkl','wb')
pickle.dump(data, output)
output.close()


