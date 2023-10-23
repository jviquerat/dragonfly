###############################################
### Class for regular off-policy agent update
class off_policy():
    def __init__(self):
        pass

    # Update
    def update(self, agent, btc_size, n_rollout):

        # Prepare a buffer of size n_rollout*btc_size
        lgt, ready = agent.prepare_data(n_rollout*btc_size)
        if (not ready): return

        # Train n_rollout times on different minibatches
        for i in range(n_rollout):
            start = i*btc_size
            end   = (i+1)*btc_size

            agent.train(start, end)
