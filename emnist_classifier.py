from __future__ import absolute_import, division, print_function

from scipy import io as spio
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

	# input layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# convolutional layer 1
	# we apply 32 5x5 filters to the input layer with retified linear unit activation function
	# use conv2d() method to create this layer
	# input_layer must have shape [batch_size, image_height, image_width, channels]

	# so convolutional layer 1 is different from input layer?
	# filters -> # of filters to apply
	# kernel_size -> dimensions of the filters
	# padding -> is either "valid" or "same", "same" means output tensor will have same height/width as input tensor
	# activation -> what activation function to apply to the output of the convolution
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# pooling layer 1
	# max_pooling2d constructs a layer that performs max pooling with 2x2 filter and stride of 2
	# pooling layer serves to reduce spatial size of the representation, to reduce number of parameters
	# and amount of computation in the network, and hence to control overfitting

	# what is max pooling?

	# pool_size -> max pooling filter size
	# strides -> size of stride, indicates that the subregions extracted by the filter should be
	# seperated by 2 pixels in both the height and width dimensions (you can declare strides as strides=[w, h])

	# output tensor produced by max_pooling2d has shape of [batch_size, 14, 14, 32] (the 2x2 filter reduces height and weight by 1/2)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# convolutional layer 2
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# pooling layer 2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# dense layer
	# flatten pool2 to shape [batch_size, features] so our tensor is 2d
	# -1 in the param means batch_size will by dynamically calculated based on # examples in input data
	# output has shape [batch_size, 3136]
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

	# units = # of neurons in dense layer
	# output has shape [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu) # relu = Retified Linear Unit

	# apply dropout regularization to dense layer to improve results
	# means % of elements will be dropped randomly during training
	# dropout will occur iff mode is TRAIN
	# output has shape [batch_size, 1024]
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# logits layer - which returns the raw values for predictions
	# output has shape [batch_size, units]
	logits = tf.layers.dense(inputs=dropout, units=27) # need 27 because images are labeled 1 -> 26 rather than 0 -> 25

	predictions = {
		# generate predicitons (for PREDICT and EVAl mode)
		"classes": tf.argmax(input=logits, axis=1), # returns predicted class corresponding to row with highest raw value
		# add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor") # get probability from logits layer by applying softmax
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# calculate loss (for TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# configure the training options (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels,
			predictions=predictions["classes"])
	}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
	# load training and eval data
	emnist = spio.loadmat("./matlab/emnist-letters.mat")

	train_data = emnist["dataset"][0][0][0][0][0][0]
	#train_data = train_data.astype(np.float32)
	train_data = np.asarray(train_data, dtype=np.float32)
	train_labels = emnist["dataset"][0][0][0][0][0][1]
	train_labels = np.asarray(train_labels, dtype=np.int32)
	#train_labels = train_labels.astype(np.int32)

	eval_data = emnist["dataset"][0][0][1][0][0][0]
	eval_data = np.asarray(eval_data, dtype=np.float32)
	#eval_data = eval_data.astype(np.float32)
	eval_labels = emnist["dataset"][0][0][1][0][0][1]
	eval_labels = np.asarray(eval_labels, dtype=np.int32)
	#eval_labels = eval_labels.astype(np.int32)

	'''
	emnist = tf.contrib.learn.datasets.load_dataset("mnist")
	train_data = emnist.train.images # returns np.array
	train_labels = np.asarray(emnist.train.labels, dtype=np.int32)
	eval_data = emnist.test.images # return np.array
	eval_labels = np.asarray(emnist.test.labels, dtype=np.int32)
	'''

	# create the estimator
	emnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn,
		model_dir="./tmp/emnist_cnn_model")

	# set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log,
		every_n_iter=50)

	# train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,
		num_epochs=None,
		shuffle=True)
	emnist_classifier.train(
		input_fn=train_input_fn,
		steps=10000,
		hooks=[logging_hook])

	# evaluate model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,
		shuffle=False)
	eval_results = emnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

	export_dir = emnist_classifier.export_savedmodel(
		export_dir_base='./models/',
		serving_input_receiver_fn=serving_input_receiver_fn()
	)

def serving_input_receiver_fn():
	inputs = {"x": tf.placeholder(shape=[None, 4], dtype=tf.float32)}
	return tf.estimator.export.ServinginputReceiver(inputs, inputs)

if __name__ == "__main__":
	tf.app.run()


# CNNs apply series of filters to raw pixel data of image to extract and learn higher-level features
# which the model then uses for classification
# CNNs contain 3 components
# convolutional layers - applies a specified number of convolution filters to image
# pooling layers - downsample image data
# dense (fully connected) layers - perform classification on the extracted features (obtained from earlier two layers)
# in dense layer, every node is connected to every node in preceding layer

# for this program, this will be the CNN architecture:
# 1. convolutional layer 1
# 2. pooling layer 1
# 3. convolutional layer 2
# 4. pooling layer 2
# 5. dense layer 1
# 6. dense layer 2 (output)
