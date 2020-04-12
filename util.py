seed = 2020
train_split = 0.8

def prep_dataset(path_KB, path_QA):
	'''
	Input:
		path_KB.txt, path_QA.txt
	Return:
		KG
		list of (q, e_s, ans)				# e_s should be replaced inside the questions also
	'''

	# TODO: Implement
	pass

def train_test_split(dataset):
	'''
	Input:
		List of (q, e_s, ans), train_split, seed
	Return:
		train_set: list of (q, e_s, ans)
		test_set: list (q, e_s, ans)
	'''
	pass

def save_checkpoint(policy_network):
	'''
	Input: PolicyNetwork
	Return: None
	Output: Appropriate save file of learned parameters weights and values for all labelled #Trainable in PolicyNetwork
			(Label file_extension according to date_time e.g. T_<T>_model_HHMM_DDMM, savedir = ./models)

	# TODO: Implement
	'''
	pass
