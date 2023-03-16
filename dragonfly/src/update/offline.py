###############################################
### Class for regular offline agent update
class offline():
    def __init__(self):
        pass

    def reset(self):
        pass

    # Update
    def update(self, agent, btc_size, n_stp):

        for i in range(n_stp):
            agent.prepare_data(btc_size)
            agent.train(btc_size)
