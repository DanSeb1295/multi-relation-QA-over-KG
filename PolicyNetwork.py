from Environment import d, State, Environment
from components import BiGRU, GRU, Perceptron, SLP, Embedder, Attention
from util import train_test_split, save_checkpoint, write_model_name
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras import utils as np_utils
from tqdm import tqdm

class PolicyNetwork(tf.keras.Model):
    def __init__(self, T, saved_model_name: str = '', env: Environment = None):
        super(PolicyNetwork, self).__init__()
        self.T = T
        self.env = env
        self.beam_size = 1
        self.lr = 1e-3
        self.ita_discount = 0.9
        self.opt = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.save_model_dir = './saved_models/'

        if saved_model_name:
            self.load_saved_model(saved_model_name)

    def call(self, x):
        q_vector, H_t = x
        return self.sub_forward(q_vector, H_t)

    def load_saved_model(self, saved_model_name):
        try:
            self.model = keras.models.load_model(self.save_model_dir + saved_model_name)
        except:
            print('Load failed. Initialise new network.')
            

    def initialise_models(self):
        self.GRU = GRU()
        self.Perceptron = Perceptron()
        self.Attention = Attention()
        self.BiGRU = BiGRU()
        self.SLP = SLP(self.T)
        self.Embedder = Embedder()


    def initialise(self):
        if not self.env:
            self.env = Environment(self.KG)

        if not hasattr(self, 'model'):
            self.model = PolicyNetwork(self.T, env=self.env)
            self.model.initialise_models()


    def train(self, inputs, epochs=10, attention=True, perceptron=True):
        KG, dataset, T = inputs
        train_set, test_set = train_test_split(dataset)

        # Hyperparameters configuration
        self.T = T
        self.KG = KG
        
        self.initialise()

        self.model.use_attention = attention
        self.model.use_perceptron = perceptron

        train_acc = []
        train_losses = []
        val_acc = []
        val_losses = []
        for i, epoch in enumerate(range(epochs)):
            print("\n>>>>>>>>>>>> EPOCH: ", i + 1, " / ", epochs)
            train_accuracy, train_loss = self.run_train_op(train_set)
            val_accuracy, val_loss = self.run_val_op(test_set)
            
            train_acc.append(train_accuracy)
            train_losses.append(train_loss)
            val_acc.append(val_accuracy)
            train_losses.append(val_loss)
            
            # Save Model
            model_name = 'model'
            model_type = 'combined'
            if attention: 
                model_name += '_att'
                model_type = 'attention'
            if perceptron: 
                model_name += '_per'
                model_type = 'perceptron'
            model_name += str(i + 1)
            
            write_model_name(model_name, model_type)
            tf.saved_model.save(self.model, self.save_model_dir + model_name)

        return (train_acc, train_losses), (val_acc, val_losses)



    def predict(self, inputs, attention=True, perceptron=True):
        KG, dataset, T = inputs
        # Hyperparameters configuration
        self.T = T
        self.KG = KG
        
        self.initialise()

        self.model.use_attention = attention
        self.model.use_perceptron = perceptron

        val_acc, predictions = self.run_val_op(dataset, predictions = True)
        return val_acc, predictions


    def run_train_op(self, train_set, predictions = False):
        # Hyperparameters configuration
        self.model.beam_size = 1
        y_hat = []
        losses = []
        
        print('\n============ TRAINING ============')
        for inputs in tqdm(train_set):
            with tf.GradientTape(persistent=True) as tape:
                prediction, outputs = self.forward(inputs)
                y_hat.append(prediction)
                if outputs is None: continue
                loss = self.REINFORCE_loss_function(outputs)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
            losses.append(loss)

        acc = np.mean([y_hat[i] == train_set[i][-1] for i in range(len(y_hat))])
        loss = np.mean(losses)
        
        return acc, loss

    def run_val_op(self, val_set, predictions = False):
        # Hyperparameters configuration
        self.model.beam_size = 32
        y_hat = []
        losses = []

        print('\n============ VALIDATING ============')
        for inputs in tqdm(val_set):
            prediction, outputs = self.forward(inputs)
            y_hat.append(prediction)
            if outputs is None: continue
            loss = self.REINFORCE_loss_function(outputs)
            losses.append(loss)

        acc = np.mean([y_hat[i] == val_set[i][-1] for i in range(len(y_hat))])
        loss = np.mean(losses)
        
        return acc, loss


    def forward(self, inputs):
        q, e_s, ans = inputs
        T = self.T

        temp_q = np.empty((0, 50)).astype(np.float32)
        for w in q:
            embeded_word = self.model.Embedder.embed_word(w)
            if embeded_word is not None and embeded_word.all():
                temp_q = np.append(temp_q, embeded_word.reshape((1,50)), axis = 0)
        q = temp_q
        q = tf.convert_to_tensor(value=q, dtype=tf.float32)     # Embedding Module
        q = tf.reshape(q, [1, *q.shape])

        r_0 = np.zeros(d).astype(np.float32)
        q_vector = self.model.bigru(q)                   # BiGRU Module
        self.model.env.start_new_query(State(q, e_s, e_s, set()), ans)
        prediction, outputs = self.model([q_vector, self.model.gru(r_0)])
        return prediction, outputs
        
        # #OUTPUTS
        # rewards = []
        # action_probs = []
        # actions_onehot = []

        # S_t = {}        # T x States; state
        # q_t = {}        # T x d x n; question
        # H_t = {}        # T x d; encoded history
        # r_t = {}        # T x d; relation
        # a_t = {}        # T x d x 2(relation, next_node)
        # q_t_star = {}     # T x d; attention weighted question

        # S_t[1] = State(q, e_s, e_s, set())
        # H_t[0] = np.zeros(d).astype(np.float32)
        # r_t[0] = np.zeros(d).astype(np.float32)
        # H_t[1] = self.gru(r_t[0])                 # History Encoder Module
        
        
        # for t in range(1, T+1):
        #     q_t[t] = self.slp(q_vector, t)             # Single-Layer Perceptron Module
        #     possible_actions = self.env.get_possible_actions()

        #     # Reached terminal node
        #     if not possible_actions: break

        #     action_space = self.beam_search(possible_actions)

        #     semantic_scores = []
        #     for action in action_space:
        #         # Attention Layer: Generate Similarity Scores between q and r and current point of attention
        #         r_star = self.Embedder.embed_relation(action[0])
        #         if r_star.all():
        #             r_star = tf.Variable(r_star)
        #             q_t_star[t] = self.attention(r_star, q_t[t])

        #             # Perceptron Module: Generate Semantic Score for action given q
        #             score = self.perceptron(r_star, H_t[t], q_t_star[t])
        #             semantic_scores.append(score)
        #         else:
        #             continue
            
        #     # Softmax Module: Leading to selection of action according to policy
        #     action_distribution = tf.nn.softmax(semantic_scores)
        #     action = self.sample_action(action_space, action_distribution)

        #     a_t[t] = action
        #     r_t[t] = self.Embedder.embed_relation(action[0])
        #     H_t[t+1] = self.gru(r_t[t])

        #     # Take action, advance state, and get reward
        #     # q_t & H_t passed in order to generate the new State object within Environment
        #     new_state, new_reward = self.env.transit(action, t, q_t, H_t)
        #     S_t[t+1] = new_state
            
        #     # Record action, state and reward
        #     # trajectory += [S_t[t], a_t[t]]
        #     #TODO: Implement discount factor
        #     rewards.append(new_reward)
        #     action_probs.append(action_distribution)
        #     actions_onehot.append(np_utils.to_categorical(np.arange(len(action_space)), num_classes=len(action_space)))

        # prediction = S_t[len(S_t)].e_t
        # discount_r = self.discount_rewards(rewards)
        # action_probs = pad_sequences(action_probs,padding='post')
        # actions_onehot = pad_sequences(actions_onehot,padding='post')

        # return prediction, [actions_onehot,action_probs,discount_r]

    def sub_forward(self, q_vector, H_t_t):
        #OUTPUTS
        rewards = []
        action_probs = []
        actions_onehot = []

        # Trajectories
        S_t = {}        # T x States; state
        q_t = {}        # T x d x n; question
        H_t = {}        # T x d; encoded history
        r_t = {}        # T x d; relation
        a_t = {}        # T x d x 2(relation, next_node)
        q_t_star = {}   # T x d; attention weighted question

        H_t[0] = np.zeros(d).astype(np.float32)
        r_t[0] = np.zeros(d).astype(np.float32)
        H_t[1] = H_t_t
        S_t[1] = self.env.current_state

        for t in range(1, self.T+1):
            q_t[t] = self.slp(q_vector, t)  # Single-Layer Perceptron Module
            possible_actions = self.env.get_possible_actions()

            # Reached terminal node
            if not possible_actions: break

            action_space = self.beam_search(possible_actions)

            semantic_scores = []
            for action in action_space:
                # Attention Layer: Generate Similarity Scores between q and r and current point of attention
                r_star = self.Embedder.embed_relation(action[0])
                if r_star is not None and r_star.all():
                    r_star = tf.Variable(r_star)

                    if self.use_attention:
                        q_t_star[t] = self.attention(r_star, q_t[t])
                    else:
                        q_t_star[t] = tf.reduce_sum(q_t[t], 0)

                    # Perceptron Module: Generate Semantic Score for action given q
                    if self.use_perceptron:
                        score = self.perceptron(r_star, H_t[t], q_t_star[t])
                    else:
                        r_star = tf.nn.l2_normalize(r_star, 0)
                        temp_q_t_star = tf.nn.l2_normalize(q_t_star[t], 0)
                        score = tf.reduce_sum(tf.math.multiply(r_star, temp_q_t_star))

                    semantic_scores.append(score)
                else:
                    continue

            if not semantic_scores: break

            # Softmax Module: Leading to selection of action according to policy
            action_distribution = tf.nn.softmax(semantic_scores)
            action = self.sample_action(action_space, action_distribution)

            a_t[t] = action
            r_t[t] = self.Embedder.embed_relation(action[0])
            H_t[t+1] = self.gru(r_t[t])

            # Take action, advance state, and get reward
            # q_t & H_t passed in order to generate the new State object within Environment
            new_state, new_reward = self.env.transit(action, t, q_t, H_t)
            S_t[t+1] = new_state

            # Record action, state and reward
            rewards.append(new_reward)
            action_probs.append(action_distribution)
            actions_onehot.append(np_utils.to_categorical(np.arange(len(action_space)), num_classes=len(action_space)))

        prediction = S_t[len(S_t)].e_t
        if not rewards:
            return prediction, None
        discount_r = self.discount_rewards(rewards)
        output = []
        for onehot, prob in zip(action_probs, actions_onehot):
          action_prob.append(tf.reduce_sum(prob * onehot, axis=1))

        return prediction, output


    def discount_rewards(self, rewards, normalize = False):
        discounted_r = np.zeros_like(rewards).astype(np.float32)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.ita_discount + rewards[t]
            discounted_r[t] = running_add

        if normalize:
            discounted_r = (discounted_r - np.mean(discounted_r)) / (np.std(discounted_r) + 1e-7)
        return discounted_r


    def REINFORCE_loss_function(self, output):
        log_action_prob = tf.math.log(tf.cast(output, dtype=tf.float32))     #Log likelihood of probabilities
        loss = - log_action_prob * rewards
        return tf.reduce_mean(loss)


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

        sorted_actions = sorted(actions_scores, key = lambda x: x[1])[:beam_size]
        beamed_actions = [action_score[0] for action_score in sorted_actions]

        return beamed_actions


    def sample_action(self, actions, probabilities):
        # Convert probabilities to log_probabilities and reshape it to [1, action_space]
        rescaled_probas = tf.expand_dims(tf.math.log(probabilities), 0)  # shape [1, action_space]

        # Draw one example from the distribution (we could draw more)
        index = tf.compat.v1.multinomial(rescaled_probas, num_samples=1)
        index = tf.squeeze(index, [0]).numpy()[0]
        
        return actions[index]
