def load_dataset():
	'''	Return dataset with info. '''
	config = configuration()

	# Retrieve percentages
	train = config.get("data_train_prc")
	val = config.get("data_val_prc")
	test = config.get("data_test_prc")

	# Redefine splits over full dataset
	splits = [f'train[:{train}%]+test[:{train}%]',\
		f'train[{train}%:{val}%]+test[{train}%:{val}%]',\
		f'train[{val}%:{test}%]+test[{val}%:{test}%]']

	# Return data
	return tfds.load('oxford_iiit_pet:3.*.*', split=splits, data_dir=configuration()\
		.get("dataset_path"), with_info=True) 

def normalize_sample(input_image, input_mask):
	''' Normalize input image and mask class. '''
	# Cast image to float32 and divide by 255
	input_image = tensorflow.cast(input_image, tensorflow.float32) / 255.0

  	# Bring classes into range [0, 2]
	input_mask -= 1

	return input_image, input_mask

def preprocess_sample(data_sample):
	''' Resize and normalize dataset samples. '''
	config = configuration()

	# Resize image
	input_image = tensorflow.image.resize(data_sample['image'],\
  	(config.get("input_width"), config.get("input_height")))

  	# Resize mask
	input_mask = tensorflow.image.resize(data_sample['segmentation_mask'],\
  	(config.get("mask_width"), config.get("mask_height")))

  	# Normalize input image and mask
	input_image, input_mask = normalize_sample(input_image, input_mask)

	return input_image, input_mask

def data_augmentation(inputs, labels):
	''' Perform data augmentation. '''
	# Use the same seed for deterministic randomness over both inputs and labels.
	seed = 36

	# Feed data through layers
	inputs = tensorflow.image.random_flip_left_right(inputs, seed=seed)
	inputs = tensorflow.image.random_flip_up_down(inputs, seed=seed)
	labels = tensorflow.image.random_flip_left_right(labels, seed=seed)
	labels = tensorflow.image.random_flip_up_down(labels, seed=seed)

	return inputs, labels

def compute_sample_weights(image, mask):
	''' Compute sample weights for the image given class. '''
	# Compute relative weight of class
	class_weights = configuration().get("class_weights")
	class_weights = class_weights/tensorflow.reduce_sum(class_weights)

  	# Compute same-shaped Tensor as mask with sample weights per
  	# mask element. 
	sample_weights = tensorflow.gather(class_weights,indices=\
  	tensorflow.cast(mask, tensorflow.int32))

	return image, mask, sample_weights

def preprocess_dataset(data, dataset_type, dataset_info):
	''' Fully preprocess dataset given dataset type. '''
	config = configuration()
	batch_size = config.get("batch_size")
	buffer_size = config.get("buffer_size")

	# Preprocess data given dataset type.
	if dataset_type == "train" or dataset_type == "val":
		# 1. Perform preprocessing
		# 2. Cache dataset for improved performance
		# 3. Shuffle dataset
		# 4. Generate batches
		# 5. Repeat
		# 6. Perform data augmentation
		# 7. Add sample weights
		# 8. Prefetch new data before it being necessary.
		return (data
				    .map(preprocess_sample)
				    .cache()
				    .shuffle(buffer_size)
				    .batch(batch_size)
				    .repeat()
				    .map(data_augmentation)
				    .map(compute_sample_weights)
				    .prefetch(buffer_size=tensorflow.data.AUTOTUNE))
	else:
		# 1. Perform preprocessing
		# 2. Generate batches
		return (data
						.map(preprocess_sample)
						.batch(batch_size))