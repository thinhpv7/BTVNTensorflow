
Mục đích: dùng để nhận dạng chữ số viết ta
#Data dimension

num_input = 28          # MNIST data input (image shape: 28x28)

timesteps = 28          # Timesteps

số class sử dụng là 10

n_classes = 10          # Number of classes, one class per digit


ảnh có kích thước là 28x28 với các giá trị từ 0 đến 1

#fixed-size image (28x28 pixels) with values from 0 to 1

#each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).


#MNIST is a dataset of handwritten digits


dữ liệu dùng huấn luyện là: 55000

#55000 examples for training

dữ liệu dùng validation là: 5000

#5000 examples for validation

dữ liệu dùng test là: 10000

#10000 examples for testing

learning_rate = 0.001 # The optimization initial learning rate

epochs = 10         # Total number of training epochs

batch_size = 100      # Training batch size

display_freq = 100    # Frequency of displaying the training results

num_hidden_units = 128  # Number of hidden units of the RNN
