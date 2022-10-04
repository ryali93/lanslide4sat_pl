def training_callbacks():
	''' Retrieve initialized callbacks for model.fit '''
	return [
		TensorBoard(
		  log_dir=os.path.join(os.getcwd(), "unet_logs"),
		  histogram_freq=1,
		  write_images=True
		)
	]

def main():
	''' Run full training procedure. '''

	# Load config
	config = configuration()
	batch_size = config.get("batch_size")
	validation_sub_splits = config.get("validation_sub_splits")
	num_epochs = config.get("num_epochs")

	# Load data
	(training_data, validation_data, testing_data), info = load_dataset()

	# Make training data ready for model.fit and model.evaluate
	train_batches = preprocess_dataset(training_data, "train", info)
	val_batches = preprocess_dataset(validation_data, "val", info)
	test_batches = preprocess_dataset(testing_data, "test", info)
	
	# Compute data-dependent variables
	train_num_samples = tensorflow.data.experimental.cardinality(training_data).numpy()
	val_num_samples = tensorflow.data.experimental.cardinality(validation_data).numpy()
	steps_per_epoch = train_num_samples // batch_size
	val_steps_per_epoch = val_num_samples // batch_size // validation_sub_splits

	# Initialize model
	model = init_model(steps_per_epoch)

	# Train the model	
	model.fit(train_batches, epochs=num_epochs, batch_size=batch_size,\
		steps_per_epoch=steps_per_epoch, verbose=1,
		validation_steps=val_steps_per_epoch, callbacks=training_callbacks(),\
		validation_data=val_batches)

	# Test the model
	score = model.evaluate(test_batches, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

	# Take first batch from the test images and plot them
	for images, masks in test_batches.take(1):

		# Generate prediction for each image
		predicted_masks = model.predict(images)

		# Plot each image and masks in batch
		for index, (image, mask) in enumerate(zip(images, masks)):
			generate_plot(image, mask, predicted_masks[index])
			if index > 4:
				break