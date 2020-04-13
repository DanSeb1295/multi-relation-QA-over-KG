from Environment import d, State, Environment
from components import BiGRU, GRU, Perceptron, SLP, Embedder
from util import train_test_split, save_checkpoint
import numpy as np
import tensorflow as tf
from tf import keras
from keras import backend as K
from tqdm import tqdm

class PolicyNetwork():
	def __init__(self, T, saved_model_path: str = ''):
		self.T = T
		self.env = None
		self.beam_size = 1
		self.lr = 1e-3

		if saved_model_path:
			self.load_saved_model(saved_model_path)
		else:
			self.initialise_models()

	def load_saved_model(self, saved_model_path):
		try:
			'''
			TODO: LOAD MODELS
				# self.Embedder = Embedder
				# self.BiGRU = BiGRU 
				# self.SLP = SLP
				# self.GRU = GRU
				# self.Perceptron = Perceptron
			'''
			pass
		except:
			self.initialise_models()

	def initialise_models(self):
		'''
		TODO:
			# self.embedder = Embedder
			# self.GRU = GRU
			# self.Perceptron = Perceptron
		'''
		self.BiGRU = BiGRU()
		self.SLP = SLP(self.T)

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

		with tf.Session() as sess:
			K.set_session(sess)
			sess.run(tf.global_variables_initializer())

			train_acc = []
			val_acc = []
			for epoch in epochs:
				epoch_train_acc = self.run_train_op(train_set)
				epoch_val_acc = self.run_val_op(test_set)
				
				train_acc.append(epoch_train_acc)
				val_acc.append(epoch_val_acc)

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
		T = self.T
		knowledge_graph = self.KG
		q_list, e_s_list, ans_list = train_set

		# TODO: Define TF loss function
		# TODO: Redo in TF
		total_reward = 0
		for q, e_s, ans in zip(q_list, e_s_list, ans_list):
			trajectory = []
			rewards = []
			q = self.embed(q)							# Embedding Module
			n = len(q)

			e_t = {}		# T x 1
			h_t = {}		# T x set()
			S_t = {}		# T x States
			q_t = {}		# T x d x n
			H_t = {}		# T x d
			r_t = {}		# T x d
			a_t = {}		# T x d x 2(relation, node)
			w_t_m = {}		# T x d
			q_t_star = {}	# T x d

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
				possible_actions = env.get_possible_actions()
				action_space = self.beam_search(possible_actions)

				semantic_scores = []
				for action in action_space:
					# Attention Layer: Generate Similarity Scores between q and r and current point of attention
					r_star = self.embed(action[0])
					q_t_star[t] = self.attention(r_star, q_t[t])

					# Perceptron Module: Generate Semantic Score for action given q
					score = self.perceptron(r_star, H_t[t], q_t_star[t])
					semantic_scores.append(score)
				
				# Softmax Module: Leading to selection of action according to policy
				action_distribution = self.generate_action_distribution(action_space, semantic_scores)
				action = self.sample_action(action_distribution)
				a_t[t] = action

				# Take action, advance state, and get reward
				# q_t & H_t passed in order to generate the new State object within Environment
				new_state, new_reward = env.transit(action, t, q_t, H_t)
				S_t[t+1] = new_state

				# Record action, state and reward
				trajectory += [S_t[t], a_t[t]]
				rewards.append(new_reward)

			total_reward += sum(rewards)

		return total_reward


	def run_val_op(self, val_set, predictions = False):
		# Hyperparameters configuration
		self.beam_size = 32
		T = self.T
		knowledge_graph = self.KG
		q, e_s, ans = val_set

		# TODO: Implement val_op
			# if predictions: return val_acc, predictions
		pass

	# TRAINABLE
	def bigru(self, q):
		# d = 100, num_layers = 2, hidden_dim = 150
		# TODO: return q_vector
		pass

	# TRAINABLE
	def slp(self, q_vector):
		# TODO: return q_t = Tanh(Wt * q_vector + b_t)
		pass

	# TRAINABLE
	def gru(self, H_t, r_t):
		# num_layers = 3, hidden_dim = 300, dropout = 0.3, Xavier Initialisation
		# TODO: return H_t_plus_1
		pass

	# TRAINABLE
	def attention(self, r_star, q_t):
		# TODO: tf.nn.softmax
		# a_star = self.softmax(b_m_star)
		# q_t_star[t] = np.dot(np.array(a_star).T, q_t[t])	#  q_t[t] give w_i_ts
		pass

	# TRAINABLE
	def perceptron(self, action, q):
		# TODO: return S(a_t, q) = r_star * W_L2 * ReLU(W_L1 * [H_t;q_t_star])
		pass

	# PRE-TRAINED
	def embed(self, vector):
		return self.Embedder.embed(vector)

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

	def generate_action_distribution(self, actions, semantic_scores):
		probabilities = self.softmax(semantic_scores)
		return sorted(list(zip(actions, probabilities)), key=lambda x: x[0][0])		# action_distribution is sorted alphabetically based on the relation
		
	def sample_action(self, action_distribution):
		# Separates list of (action, prob) to actions list and probs list
		actions, probs = list(zip(*action_distribution))
		sampled_index = np.random.choice(len(actions), p=probs)
		return actions[sampled_index]

	def softmax(self, vectors):
		return np.exp(vectors) / np.sum(np.exp(vectors))
