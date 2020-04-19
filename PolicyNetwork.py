from Environment import d, State, Environment
from components import BiGRU, GRU, Perceptron, SLP, Embedder, Attention
from util import train_test_split, save_checkpoint
import numpy as np
import tensorflow as tf
from tf import keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
from tqdm import tqdm


class PolicyNetwork():
	def __init__(self, T, saved_model_path: str = ''):
		self.T = T
		self.env = None
		self.beam_size = 1
		self.lr = 1e-3
		self.ita_discount = 0.8
		self.opt = tf.train.AdamOptimizer(learning_rate = self.lr)
		self.sess = tf.Session()

		if saved_model_path:
			self.load_saved_model(saved_model_path)
		else:
			self.initialise_models()

	def load_saved_model(self, sess, saved_model_path):
		self.initialise_models()
		try:
			saver = tf.train.import_meta_graph(saved_model_path)
			saver.restore(self.sess, tf.train.latest_checkpoint('./'))
		except:
			print('no load file, starting from scratch')
			

	def initialise_models(self):
		self.GRU = GRU()
		self.Perceptron = Perceptron
		self.Attention = Attention()
		self.BiGRU = BiGRU()
		self.SLP = SLP(self.T)
		self.Embedder = Embedder


	def train(self, inputs, epochs=10, attention=True, perceptron=True):
		KG, dataset, T = inputs
		train_set, test_set = train_test_split(dataset)
		
		# Hyperparameters configuration
		self.T = T
		self.KG = KG
		self.use_attention = attention
		self.use_perceptron = perceptron
		if not self.env:
			self.env = Environment(KG)


		with self.sess:
			K.set_session(self.sess)
			self.sess.run(tf.global_variables_initializer())

			train_acc = []
			val_acc = []
			for epoch in range(epochs):
				epoch_train_acc = self.run_train_op(train_set)
				epoch_val_acc = self.run_val_op(test_set)
				
				train_acc.append(epoch_train_acc)
				val_acc.append(epoch_val_acc)
				
				# save results and weights
				with open("results.txt", "a+") as f:
					f.write("Iteration %s - train acc: %d, val acc: %d" % (epoch, epoch_train_acc, epoch_val_acc))
				if epoch == 1:
					save_checkpoint(self, 'model', epoch,write_meta_graph=True)
				else:
					save_checkpoint(self,'model',epoch)

		return train_acc, val_acc

	def predict(self, inputs, attention=True, perceptron=True):
		KG, dataset, T = inputs
		# Hyperparameters configuration
		self.T = T
		self.KG = KG
		self.use_attention = attention
		self.use_perceptron = perceptron
		if not self.env:
			self.env = Environment(KG)

		val_acc, predictions = self.run_val_op(dataset, predictions = True)
		return val_acc, predictions


	def run_train_op(self, train_set):
		# Hyperparameters configuration
		self.beam_size = 1
		for inputs in train_set:
			predictions, outputs = self.forward(inputs)
			loss = self.REINFORCE_loss_function(outputs)
			self.opt.minimize(loss)
		return loss

	def run_val_op(self, val_set, predictions = False):
	        # Hyperparameters configuration
	        self.beam_size = 32
	        T = self.T
	        n = len(val_set)
	        y_hat = []

	        for inputs in val_set:
	            predictions, outputs = self.forward(inputs)
	            y_hat.append(y_pred)
	        if predictions:
	            acc = np.mean([y_hat[i] == val_set[i][-1] for i in range(n)])
	        return acc, y_hat


	def forward(self, inputs):
		q, e_s, ans = inputs
		T = self.T

		#OUTPUTS
		rewards = []
		action_probs = []
		actions_onehot = []
		
		q = [self.Embedder.embed_word(w) for w in q]
		q = tf.convert_to_tensor(q)							# Embedding Module
		n = len(q)

		e_t = {}		# T x 1; entity
		h_t = {}		# T x set(); history
		S_t = {}		# T x States; state
		q_t = {}		# T x d x n; question
		H_t = {}		# T x d; encoded history
		r_t = {}		# T x d; relation
		a_t = {}		# T x d x 2(relation, next_node)
		w_t_m = {}		# T x d; attention distribution
		q_t_star = {}	# T x d; attention weighted question

		e_t[1] = e_s
		h_t[1] = set()		# OR LIST????
		S_t[1] = State(q, e_s, e_t[1], h_t[1])
		q_vector = self.bigru(q)					# BiGRU Module
		H_t[0] = np.zeros(d)
		r_t[0] = np.zeros(d)
		
		self.env.start_new_query(S_t[1], ans)
		
		for t in range(1, T+1):
			q_t[t] = self.slp(q_vector)				# Single-Layer Perceptron Module
			H_t[t] = self.gru(H_t[t-1], r_t[t-1])	# History Encoder Module
			possible_actions = self.env.get_possible_actions()
			action_space = self.beam_search(possible_actions)

			semantic_scores = []
			for action in action_space:
				# Attention Layer: Generate Similarity Scores between q and r and current point of attention
				r_star = self.Embedder.embed_relation(action[0])
				q_t_star[t] = self.attention(r_star, q_t[t])

				# Perceptron Module: Generate Semantic Score for action given q
				score = self.perceptron(r_star, H_t[t], q_t_star[t])
				semantic_scores.append(score)
			
			# Softmax Module: Leading to selection of action according to policy
			action_prob = self.softmax(semantic_scores)
			action = self.sample_action(action_distribution)
			a_t[t] = action

			# Take action, advance state, and get reward
			# q_t & H_t passed in order to generate the new State object within Environment
			new_state, new_reward = self.env.transit(action, t, q_t, H_t)
			S_t[t+1] = new_state
			
			# Record action, state and reward
			trajectory += [S_t[t], a_t[t]]
			#TODO: Implement discount factor
			rewards.append(new_reward)
			action_probs.append(action_prob)
			actions_onehot.append(np_utils.to_categorical(action, num_classes=len(action_prob)))

		prediction = S_t[-1].e_t
		discount_r = self.discount_rewards(rewards)
		action_probs = pad_sequences(action_probs,padding='post')
		actions_onehot = pad_sequences(actions_onehot,padding='post')

		return prediction, [actions_onehot,action_probs,discount_r]


	def discount_rewards(self, rewards):
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.ita_discount + rewards[t]
            discounted_r[t] = running_add
		discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-7)
        return discounted_r


	def REINFORCE_loss_function(self, outputs):
		actions_onehot, action_probs, rewards = outputs
		action_prob = K.sum(action_probs * actions_onehot, axis=1)
		log_action_prob = K.log(action_prob)
		loss = - log_action_prob * rewards
		return K.mean(loss)


	# TRAINABLE
	def bigru(self, q):
		# Returns: q_vector
		return self.BiGRU.compute(q)

	# TRAINABLE
	def slp(self, q_vector, t):
		# Returns: q_t = Tanh(Wt * q_vector + b_t)
		return self.SLP.compute(q_vector, t)

	# TRAINABLE
	def gru(self, r_t):
		# Returns: H_t_plus_1 = GRU(H_t, r_t)
		return self.GRU.compute(r_t)		

	# TRAINABLE
	def attention(self, r_star, q_t):
		# Returns: q_t_star[t]
		return self.Attention.compute(r_star, q_t)

	# TRAINABLE
	def perceptron(self, r_star, H_t, q_t_star):
		# Returns: S(a_t, q) = r_star * W_L2 * ReLU(W_L1 * [H_t; q_t_star])
		return self.Perceptron.compute(r_star, H_t, q_t_star)

	def beam_search(self, possible_actions, beam_size = None):
		if not beam_size:
			beam_size = self.beam_size
		
		actions_scores = []
		for action in possible_actions:
			expected_reward = self.env.get_action_reward(action)
			actions_scores.append((action, expected_reward))

		sorted_actions = sorted(actions_scores, key = lambda x: x[1])
		beamed_actions = [action_score[0] for action_score in sorted_actions]

		return beamed_actions

		
	def sample_action(self, actions, probs):
		sampled_index = np.random.choice(len(actions), p=probs)
		return actions[sampled_index]


	def softmax(self, vectors):
		return np.exp(vectors) / np.sum(np.exp(vectors))
