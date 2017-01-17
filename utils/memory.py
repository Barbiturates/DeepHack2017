import numpy as np

BATCH_SIZE = 32
MEMORY_SIZE = 1000000 # 10 ** 5
IMG_HEIGHT = 84
IMG_WIDTH = 84
LEARNING_SEQ_LEN = 4


class Memory:
    def __init__(self, size, batch_size=BATCH_SIZE):
        self.size = size
        self.batch_size = batch_size
        self.states = np.zeros((MEMORY_SIZE, IMG_HEIGHT, IMG_WIDTH))
        self.next_states = np.zeros((MEMORY_SIZE, IMG_HEIGHT, IMG_WIDTH))
        self.actions = np.zeros((MEMORY_SIZE,))
        self.rewards = np.zeros((MEMORY_SIZE,))
        self.is_done = np.zeros((MEMORY_SIZE,))
        self.mem_ptr = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.mem_ptr, :, :] = state
        self.next_states[self.mem_ptr, :, :] = next_state
        self.actions[self.mem_ptr] = action
        self.rewards[self.mem_ptr] = reward
        self.is_done[self.mem_ptr] = int(done)
        self.mem_ptr = (self.mem_ptr + 1) % MEMORY_SIZE

    def get_batch(self):
        if self.mem_ptr > self.batch_size:
            idx = np.random.randint(LEARNING_SEQ_LEN - 1, self.mem_ptr - 1, self.batch_size)
            states_batch = np.zeros((self.batch_size, LEARNING_SEQ_LEN, IMG_HEIGHT, IMG_WIDTH))
            next_states_batch = np.zeros((self.batch_size, LEARNING_SEQ_LEN, IMG_HEIGHT, IMG_WIDTH))
            actions_batch = np.zeros((self.batch_size,))
            rewards_batch = np.zeros((self.batch_size,))
            is_done_batch = np.zeros((self.batch_size,))

            for i in range(len(idx)):
                states_batch[i, :, :, :] = self.states[idx[i] - LEARNING_SEQ_LEN:idx[i]]
                next_states_batch[i, :, :, :] = self.next_states[idx[i] - LEARNING_SEQ_LEN:idx[i]]
                actions_batch[i] = self.actions[idx[i]]
                rewards_batch[i] = self.rewards[idx[i]]
                is_done_batch[i] = self.is_done[idx[i]]

            return states_batch, actions_batch, rewards_batch, next_states_batch, is_done_batch
