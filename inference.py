import unet
from utils import plot_input_prediction

# LOAD THE MODEL & WEIGHTS
net = unet.UNet()
net.load_model('./trained_model/unet.h5')

# LOAD THE TEST DATA
test_file = './data/test/train_1.png'

# INFERENCE AND FIGURES
net.load_testset(test_file)

img = next(iter(net.testset.take(1)))
res = net.model.predict(img.numpy().reshape(1, net.imwidth, net.imheight, 1))

plot_input_prediction(img, res, 'Result.eps')
