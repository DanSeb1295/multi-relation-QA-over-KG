import numpy as np
from numpy.linalg import norm

d = 50

class State():
    def __init__(self, q, e_s, e_t, h_t: set, t=1, q_t = [np.zeros(d)], H_t = np.zeros(d)):
        self.q = q
        self.e_s = e_s
        self.e_t = e_t
        self.h_t = h_t
        self.memory = {
            't': t,
            'q_t': q_t,
            'H_t': H_t
        }

    def set_memory(self, t, q_t, H_t):
        self.memory['t'] = t
        self.memory['q_t'] = q_t
        self.memory['H_t'] = H_t
        return self

    def get_memory(self):
        return self.memory


class Rewards():
    def __init__(self, gamma = 0.95, rewards_dict = dict()):
        self.gamma = gamma
        self.rewards_dict = rewards_dict

    def get_transition_reward(self, cur_state, action, next_state, ans):
        SAS = (cur_state, action, next_state)
        if self.rewards_dict.get(SAS): return self.rewards_dict.get(SAS)    # Skip computation if reward has previously been computed
        
        F_phi = self.gamma * self.phi(next_state) - self.phi(cur_state)
        R = 1 if next_state.e_t == ans else 0
        reward = R + F_phi
        self.rewards_dict[SAS] = reward     # Cache the (SAS -> R) mapping
        return reward

    def phi(self, state):
        state_memory = state.get_memory()
        t = state_memory.get('t')
        
        if t == 1: return 0
        
        H_t = state_memory.get('H_t')
        q_t = state_memory.get('q_t')
        Q_t = np.zeros(d)
        n = len(q.get(t, []))
        for i in range(1, t):
            for j in range(n):
                Q_t += q_t[i][j]
        
        norm_product = (norm(Q_t) * norm(H_t))
        if norm_product == 0:
            cos_sim = 0
        else:
            cos_sim = np.dot(Q_t, H_t) / norm_product

        return max(cos_sim, 0)


class Environment():
    def __init__(self, knowledge_graph, start_state: State = None, rewards_func: Rewards = None):
        self.knowledge_graph = knowledge_graph
        self.start_state = start_state
        self.current_state = start_state
        if not rewards_func:
            self.rewards_func = Rewards()

    def transit(self, action, t, q_t, H_t):
        new_state = self.get_next_state(action)
        new_state.set_memory(t+1, q_t, H_t)
        new_reward = self.get_transition_reward(self.current_state, action, new_state)
        self.current_state = new_state
        return new_state, new_reward

    def start_new_query(self, state: State, ans):
        self.start_state = state
        self.current_state = state
        self.ans = ans
        return self

    def get_next_state(self, action):
        # action = (relation, node)
        assert self.knowledge_graph.has_edge(self.current_state.e_t, action[1]), "Transition Error: This action does not exist for current node: {}".format(self.current_state)

        for neighbour, edge in self.knowledge_graph[self.current_state.e_t].items():
            if edge['relation'] == action[0]:
                new_state = State(self.current_state.q, self.current_state.e_s, neighbour, set(self.current_state.h_t) | {action})
                break
        
        return new_state

    def get_possible_actions(self):
        assert self.knowledge_graph.has_node(self.current_state.e_t), "State Error: No such node in graph"

        action_space = [(edge['relation'], neighbour) for neighbour, edge in self.knowledge_graph[self.current_state.e_t].items()]
        return action_space

    def get_action_reward(self, action):
        cur_state = self.current_state
        next_state = self.get_next_state(action)
        return self.rewards_func.get_transition_reward(cur_state, action, next_state, self.ans)

    def get_transition_reward(self, s_0: State, action, s_1: State):
        return self.rewards_func.get_transition_reward(s_0, action, s_1, self.ans)

