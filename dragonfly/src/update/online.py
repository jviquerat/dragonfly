###############################################
### Class for regular online agent update
class online():
    def __init__(self):
        pass

    # Update
    def update(self, agent, size, btc_size, n_epochs):

        for epoch in range(n_epochs):
            # Prepare training data
            lgt, ready = agent.prepare_data(4000)
            if (not ready): return

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                agent.train_mu(start, end)

                btc += 1
                if (end == lgt): done = True

        for epoch in range(4*n_epochs):
            # Prepare training data
            lgt, ready = agent.prepare_data(size)
            if (not ready): return

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                agent.train_cov(start, end)

                btc += 1
                if (end == lgt): done = True
