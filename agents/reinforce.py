import numpy as np
import agents.base
from models import model


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

    def act(self, state, *args, **kwargs):
        pass

    def react(self, state, action, reward, done, new_state, *args, **kwargs):
        pass
