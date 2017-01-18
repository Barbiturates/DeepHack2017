import numpy as np

BATCH_SIZE = 32
MEMORY_SIZE = 10 ** 4
IMG_HEIGHT = 84
IMG_WIDTH = 84
LEARNING_SEQ_LEN = 4


class Memory:
    def __init__(self, size=MEMORY_SIZE, batch_size=BATCH_SIZE):
        self.size = size
        self.batch_size = batch_size
        self.states = np.zeros((MEMORY_SIZE, IMG_HEIGHT, IMG_WIDTH))
        self.next_states = np.zeros((MEMORY_SIZE, IMG_HEIGHT, IMG_WIDTH))
        self.actions = np.zeros((MEMORY_SIZE,))
        self.rewards = np.zeros((MEMORY_SIZE,))
        self.is_done = np.zeros((MEMORY_SIZE,))
        self.mem_ptr = 0
        self.n_examples = 0

        self.episode_states_idxs = []
        self.min_cur_reward = 0
        self.min_final_reward = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.mem_ptr, :, :] = state
        self.next_states[self.mem_ptr, :, :] = next_state
        self.actions[self.mem_ptr] = action
        self.rewards[self.mem_ptr] = reward
        self.is_done[self.mem_ptr] = int(done)
        self.episode_states_idxs.append(self.mem_ptr)
        self.mem_ptr = (self.mem_ptr + 1) % MEMORY_SIZE
        self.n_examples = np.min([MEMORY_SIZE, self.n_examples + 1])

        self.min_cur_reward = min(reward, self.min_cur_reward)

    def add_final_reward(self, final_reward):
        self.min_final_reward = min(final_reward, self.min_final_reward)
        self.rewards[self.episode_states_idxs] +=\
            float(self.min_cur_reward) * (float(final_reward) / float(self.min_final_reward))
        self.episode_states_idxs.clear()

    def get_last_n(self, n=3):
        assert self.n_examples >= n
        if self.mem_ptr < n:
            return np.concatenate((self.states[-(n - self.mem_ptr):], self.states[0:self.mem_ptr]))
        return self.states[self.mem_ptr-n:self.mem_ptr]

    def get_batch(self, random=True):
        if self.mem_ptr > self.batch_size or not random:
            batch_length = LEARNING_SEQ_LEN
            if random:
                idx = np.random.randint(LEARNING_SEQ_LEN, self.mem_ptr - 1, self.batch_size)
            else:
                idx = self.mem_ptr
                batch_length -= 1
            states_batch = np.zeros((self.batch_size, batch_length, IMG_HEIGHT, IMG_WIDTH))
            next_states_batch = np.zeros((self.batch_size, batch_length, IMG_HEIGHT, IMG_WIDTH))
            actions_batch = np.zeros((self.batch_size,))
            rewards_batch = np.zeros((self.batch_size,))
            is_done_batch = np.zeros((self.batch_size,))

            for i in range(len(idx)):
                states_batch[i, :, :, :] = self.states[idx[i] - batch_length:idx[i]]
                next_states_batch[i, :, :, :] = self.next_states[idx[i] - batch_length:idx[i]]
                actions_batch[i] = self.actions[idx[i]]
                rewards_batch[i] = self.rewards[idx[i]]
                is_done_batch[i] = self.is_done[idx[i]]

            states_batch = np.transpose(states_batch, axes=(0, 2, 3, 1))
            next_states_batch = np.transpose(next_states_batch, axes=(0, 2, 3, 1))

            return states_batch, actions_batch, rewards_batch, next_states_batch, is_done_batch
