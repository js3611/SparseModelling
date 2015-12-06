import pickle

import SparseModelling.data.ReadMNIST as mnist

data = mnist.read_mnist_data('test')[0:100].reshape(100,28*28).T

output = open('sample_data.pkl','wb')
pickle.dump(data, output)
output.close()


