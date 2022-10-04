def probs_to_mask(probs):
	''' Convert Softmax output into mask. '''
	pred_mask = tensorflow.argmax(probs, axis=2)
	return pred_mask

def generate_plot(img_input, mask_truth, mask_probs):
	''' Generate a plot of input, truthy mask and probability mask. '''
	fig, axs = plt.subplots(1, 4)
	fig.set_size_inches(16, 6)

	# Plot the input image
	axs[0].imshow(img_input)
	axs[0].set_title("Input image")

	# Plot the truthy mask
	axs[1].imshow(mask_truth)
	axs[1].set_title("True mask")

	# Plot the predicted mask
	predicted_mask = probs_to_mask(mask_probs)
	axs[2].imshow(predicted_mask)
	axs[2].set_title("Predicted mask")

	# Plot the overlay
	config = configuration()
	img_input_resized = tensorflow.image.resize(img_input, (config.get("mask_width"), config.get("mask_height")))
	axs[3].imshow(img_input_resized)
	axs[3].imshow(predicted_mask, alpha=0.5)
	axs[3].set_title("Overlay")

	# Show the plot
	plt.show()