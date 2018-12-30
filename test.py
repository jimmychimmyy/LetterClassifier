from scipy import io as spio


emnist = spio.loadmat("./matlab/emnist-letters.mat")
train_data = emnist["dataset"][0][0][0][0][0][0]

print(train_data)
print(train_data.shape)
