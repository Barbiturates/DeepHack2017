import numpy as np
import agents.base
from models import model


# Number of past events to store results about
MEMORY_SIZE = 10 ** 5
IMG_WIDTH = 84
IMG_HEIGHT = 84
LEARNING_SEQ_LEN = 4
EPSILON = 1e-1


class Reinforce(agents.base.Agent):

    def __init__(self,
                 learning=True,
                 n_history=5,
                 discount=0.99,
                 iteration_size=75,
                 batch_size=32):
        super(Reinforce, self).__init__()
        self.discount = discount
        self.iteration_size = iteration_size
        self.batch_size = batch_size
        self.model = model.get_model_deepmind()

    def act(self, state, *args, **kwargs):
        state = np.expand_dims(state, 0)
        assert len(state.shape) == 4
        if np.random.rand() < EPSILON:
            return np.random.randint(0, 3, 1)
        else:
            return np.argmax(self.model.predict(state))

    def react(self, batch, *args, **kwargs):
        states, actions, rewards, new_states, dones = batch

        # feed-forward pass for new states to get Q-values
        postq = self.model.predict(new_states, self.batch_size)

        # calculate max Q-value for each new state
        maxpostq = np.max(postq, axis=0)

        # feed-forward pass for states
        preq = self.model.predict(states, self.batch_size)

        # collect targets
        targets = preq.copy()
        for i, action in enumerate(actions):
            action = int(action)
            if not dones[i]:
                targets[i, int(action)] = rewards[i] + self.discount * maxpostq[int(action)]
            else:
                targets[i, int(action)] = rewards[i]

        # back-propagation pass for states and targets
        self.model.train_on_batch(states, targets)

        return postq
