###############################################
### Class for regular online agent update
class online():
    def __init__(self):
        pass

    def reset(self):
        pass

    # Update
    def update(self, agent, n_buff, size, btc_size, n_epochs):

        for epoch in range(n_epochs):
            # Prepare training data
            lgt = agent.prepare_data(size)

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                agent.train(start, end)

                btc += 1
                if (end == lgt): done = True
