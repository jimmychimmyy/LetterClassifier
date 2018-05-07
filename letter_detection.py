# this program detects letters in images and creates a bounding box around each letter
import os
import cv2
import numpy as np
import tesserocr as tr
from PIL import Image, ImageOps
import string
import tensorflow as tf
import scipy

import emnist_classifier as clf

tf.logging.set_verbosity(tf.logging.INFO)

def main():
	image = cv2.imread('helloworld.png')

	pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	api = tr.PyTessBaseAPI()

	lower_case_letters = list(string.ascii_lowercase)
	upper_case_letters = list(string.ascii_uppercase)
	letters = list(string.ascii_letters)

	index = 0
	character = 0

	try:
		api.SetImage(pil_image)
		boxes = api.GetComponentImages(tr.RIL.SYMBOL, True)
		text = api.GetUTF8Text()

		for (im, box, _, _) in boxes:
			x, y, w, h = box['x'], box['y'], box['w'], box['h']
			cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 0, 0), thickness=1)

			sub_image = image[y:y+h, x:x+w]

			sub_image = cv2.resize(sub_image, (20, 20))

			row, col= sub_image.shape[:2]
			bottom= sub_image[row-2:row, 0:col]
			mean= cv2.mean(bottom)[0]
			bordersize=4

			border=cv2.copyMakeBorder(
				sub_image, 
				top=bordersize, 
				bottom=bordersize, 
				left=bordersize, 
				right=bordersize, 
				borderType= cv2.BORDER_CONSTANT, 
				value=[mean,mean,mean] )

			cv2.imwrite("./data/" + str(index) + ".png", border)

			# want to get every image bounded by rectangle
			# resize to 28x28
			# save each into numpy array
			# inside finally, make predictions

			print(text[character])

			character+=1
			image_resized = cv2.resize(image, (960, 540)) 
			cv2.imshow('2', image_resized)

			index+=1

	finally:
		#cv2.waitKey(0)
		api.End()

	classify(convert_handwritten_image_to_emnist_format())


		
def convert_handwritten_image_to_emnist_format():
	handwritten_dataset = os.listdir("./data/")
	num_images = len(handwritten_dataset)

	for file in os.listdir("./data/"):
		if file.startswith('.'):
			num_images = num_images - 1

	data = np.zeros((num_images, 784)) # 28x28 = 784

	index = 0
	for file in os.listdir("./data/"):
		if not file.startswith('.'):
			img = Image.open("./data/"+file)
			arr = np.array(img)
			try:
				arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
			except:
				print("could not convert image to grayscale")
			flat_arr = arr.ravel()
			data[index, :] = flat_arr
			#vector = np.matrix(flat_arr)
			#print(vector.shape)
			index+=1

	return data


def classify(data):
	with tf.Session() as sess:
		checkpoint = tf.train.latest_checkpoint('./tmp/emnist_cnn_model/')
		saver = tf.train.import_meta_graph("./tmp/emnist_cnn_model/model.ckpt-12000.meta")
		saver.restore(sess, tf.train.latest_checkpoint('./tmp/emnist_cnn_model/'))

		data = data.reshape(data.shape[0], 28, 28, 1)
		pred_data = np.asarray(data, dtype=np.float32)
		#img = tf.placeholder(shape=[len(data), 28, 28, 1], dtype=tf.float32)
		#data = tf.convert_to_tensor(images, dtype=tf.float32)
		#feed_dict = {"x": pred_data}
		print("model restored")

		emnist_classifier = tf.estimator.Estimator(
			model_fn=clf.cnn_model_fn,
			model_dir="./tmp/emnist_cnn_model")

		pred_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": pred_data},
			num_epochs=1,
			shuffle=False)

		pred_results = emnist_classifier.predict(input_fn=pred_input_fn)
		print(list(pred_results))





if __name__ == "__main__":
	tf.reset_default_graph()
	tf.logging.set_verbosity(tf.logging.INFO)
	main()
