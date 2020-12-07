import numpy as np

class Metropolis:

    def __init__(self, f, f_sampler, pi, initial_state=None):

        self.f = f
        self.f_sampler = f_sampler
        self.pi = pi
        self.state = np.random.normal() if initial_state is None else initial_state
        self.samples = []

        self.step = 0

    def q(self, x, y):
        if abs(y) <= abs(x):
            return self.f(y/x) / (2*x)
        else:
            return self.f(x/y) * x*x / (2 * x * y*y)

    def iterate(self):

        bernoulli = np.random.choice([0, 1])
        epsilon = self.f_sampler()
        proposal = epsilon * self.state if bernoulli == 1 else self.state / epsilon
        q_xy = self.q(self.state, proposal)
        q_yx = self.q(proposal, self.state)
        alpha = min(1, self.pi(proposal) * q_yx/(self.pi(self.state) * q_xy))
        U = np.random.uniform()
        if U <= alpha:
            self.state = proposal
        self.samples.append(self.state)

        self.step += 1

    def run(self, n_steps):
        for step in range(n_steps):
            self.iterate()
        return self.samples
