import numpy as np
from tensorflow.keras import Model

def full_network_embedding(model, imgs, batch_size, target_layer_names, stats=None):
	print('Generating a full network embedding for ' + str(len(target_layer_names)) + ' layers')
	# define feature extractor
	feature_extractor = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers if layer.name in target_layer_names])
	get_raw_features = lambda x: [t.numpy() for t in feature_extractor(x)]
	# prepare output variable
	feature_shapes = [layer.output_shape for layer in model.layers if layer.name in target_layer_names]
	len_features = sum(shape[-1] for shape in feature_shapes)
	features = np.empty((len(imgs), len_features))
	# extract features
	for idx in range(0, len(imgs), batch_size):
		batched_imgs = imgs[idx:idx + batch_size]
		feature_vals = get_raw_features(batched_imgs)
		features_current = np.empty((len(batched_imgs), 0))
		for feat in feature_vals:
			# if its not a conv layer, add without pooling
			if len(feat.shape) != 4:
				features_current = np.concatenate((features_current, feat), axis=1)
				continue
			# if its a conv layer, do SPATIAL AVERAGE POOLING
			pooled_vals = np.mean(np.mean(feat, axis=2), axis=1)
			features_current = np.concatenate((features_current, pooled_vals), axis=1)
		# store in position
		features[idx:idx+len(batched_imgs)] = features_current.copy()
	if stats is None:
		stats = np.zeros((2, len_features))
		stats[0, :] = np.mean(features, axis=0)
		stats[1, :] = np.std(features, axis=0)
	# apply statistics, avoiding nans after division by zero
	features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features), where=stats[1] != 0)
	if len(np.argwhere(np.isnan(features))) != 0:
		raise Exception('There are nan values after standardization!')
	# discretization step
	th_pos = 0.15
	th_neg = -0.25
	features[features > th_pos] = 1
	features[features < th_neg] = -1
	features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

	return features, stats
