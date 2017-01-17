import numpy as np
import agents.base
from models import model


# Number of past events to store results about
MEMORY_SIZE = 10 ** 5
IMG_WIDTH = 84
IMG_HEIGHT = 84
LEARNING_SEQ_LEN = 4


class Reinforce(agents.base.Agent):

    def __init__(self,
                 learning=True,
                 n_history=5,
                 discount=0.99,
                 iteration_size=75,
                 batch_size=3000):
        super(Reinforce, self).__init__()
        self.discount = discount
        self.iteration_size = iteration_size
        self.batch_size = batch_size
        self.model = model.get_model_deepmind()

        self.state_mem = np.zeros((MEMORY_SIZE, IMG_HEIGHT, IMG_WIDTH))
        self.action_mem = np.zeros((MEMORY_SIZE, ))
        self.reward_mem = np.zeros((MEMORY_SIZE, ))
        self.mem_ptr = 0

    def act(self, state, *args, **kwargs):
        return np.argmax(self.model.predict(state))

    def react(self, state, action, reward, done, new_state, *args, **kwargs):
        self.state_mem[self.mem_ptr, :, :] = state
        self.action_mem[self.mem_ptr] = action
        self.reward_mem[self.mem_ptr] = reward
        self.mem_ptr = (self.mem_ptr + 1) % MEMORY_SIZE

        if self.mem_ptr > LEARNING_SEQ_LEN:
            idx = np.random.randint(LEARNING_SEQ_LEN-1, self.mem_ptr-1, self.batch_size)
            image_batch = np.zeros((self.batch_size, LEARNING_SEQ_LEN, IMG_HEIGHT, IMG_WIDTH))
            for i in range(LEARNING_SEQ_LEN):
                image_batch[:, i, :, :] = self.state_mem[idx-i]
            
            pass
