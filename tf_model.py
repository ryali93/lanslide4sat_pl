'''
	U-NET CONFIGURATION
'''
def configuration():
	''' Get configuration. '''

	return dict(
		data_train_prc = 80,
		data_val_prc = 90,
		data_test_prc = 100,
		num_filters_start = 64,
		num_unet_blocks = 3,
		num_filters_end = 3,
		input_width = 100,
		input_height = 100,
		mask_width = 60,
		mask_height = 60,
		input_dim = 3,
		optimizer = Adam,
		loss = SparseCategoricalCrossentropy,
		initializer = HeNormal(),
		batch_size = 50,
		buffer_size = 50,
		num_epochs = 25,
		metrics = ['accuracy'],
		dataset_path = os.path.join(os.getcwd(), 'data'),
		class_weights = tensorflow.constant([1.0, 1.0, 2.0]),
		validation_sub_splits = 5,
		lr_schedule_percentages = [0.2, 0.5, 0.8],
		lr_schedule_values = [3e-4, 1e-4, 1e-5, 1e-6],
		lr_schedule_class = schedules.PiecewiseConstantDecay
	)


'''
	U-NET BUILDING BLOCKS
'''

def conv_block(x, filters, last_block):
	'''
		U-Net convolutional block.
		Used for downsampling in the contracting path.
	'''
	config = configuration()

	# First Conv segment
	x = Conv2D(filters, (3, 3),\
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Second Conv segment
	x = Conv2D(filters, (3, 3),\
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Keep Conv output for skip input
	skip_input = x

	# Apply pooling if not last block
	if not last_block:
		x = MaxPool2D((2, 2), strides=(2,2))(x)

	return x, skip_input


def contracting_path(x):
	'''
		U-Net contracting path.
		Initializes multiple convolutional blocks for 
		downsampling.
	'''
	config = configuration()

	# Compute the number of feature map filters per block
	num_filters = [compute_number_of_filters(index)\
			for index in range(config.get("num_unet_blocks"))]

	# Create container for the skip input Tensors
	skip_inputs = []

	# Pass input x through all convolutional blocks and
	# add skip input Tensor to skip_inputs if not last block
	for index, block_num_filters in enumerate(num_filters):

		last_block = index == len(num_filters)-1
		x, skip_input = conv_block(x, block_num_filters,\
			last_block)

		if not last_block:
			skip_inputs.append(skip_input)

	return x, skip_inputs


def upconv_block(x, filters, skip_input, last_block = False):
	'''
		U-Net upsampling block.
		Used for upsampling in the expansive path.
	'''
	config = configuration()

	# Perform upsampling
	x = Conv2DTranspose(filters//2, (2, 2), strides=(2, 2),\
		kernel_initializer=config.get("initializer"))(x)
	shp = x.shape

	# Crop the skip input, keep the center
	cropped_skip_input = CenterCrop(height = x.shape[1],\
		width = x.shape[2])(skip_input)

	# Concatenate skip input with x
	concat_input = Concatenate(axis=-1)([cropped_skip_input, x])

	# First Conv segment
	x = Conv2D(filters//2, (3, 3),
		kernel_initializer=config.get("initializer"))(concat_input)
	x = Activation("relu")(x)

	# Second Conv segment
	x = Conv2D(filters//2, (3, 3),
		kernel_initializer=config.get("initializer"))(x)
	x = Activation("relu")(x)

	# Prepare output if last block
	if last_block:
		x = Conv2D(config.get("num_filters_end"), (1, 1),
			kernel_initializer=config.get("initializer"))(x)

	return x


def expansive_path(x, skip_inputs):
	'''
		U-Net expansive path.
		Initializes multiple upsampling blocks for upsampling.
	'''
	num_filters = [compute_number_of_filters(index)\
			for index in range(configuration()\
				.get("num_unet_blocks")-1, 0, -1)]

	skip_max_index = len(skip_inputs) - 1

	for index, block_num_filters in enumerate(num_filters):
		skip_index = skip_max_index - index
		last_block = index == len(num_filters)-1
		x = upconv_block(x, block_num_filters,\
			skip_inputs[skip_index], last_block)

	return x


def build_unet():
	''' Construct U-Net. '''
	config = configuration()
	input_shape = (config.get("input_height"),\
		config.get("input_width"), config.get("input_dim"))

	# Construct input layer
	input_data = Input(shape=input_shape)

	# Construct Contracting path
	contracted_data, skip_inputs = contracting_path(input_data)

	# Construct Expansive path
	expanded_data = expansive_path(contracted_data, skip_inputs)

	# Define model
	model = Model(input_data, expanded_data, name="U-Net")

	return model
