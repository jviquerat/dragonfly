###############################################
### Class for mixed update
class mixed():
    def __init__(self):
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

                agent.train_actor(start, end)

                btc += 1
                if (end == lgt): done = True

        n_buff = 4
        for epoch in range(n_buff*n_epochs):
            # Prepare training data
            lgt = agent.prepare_data(n_buff*size)

            # Visit all available history
            done = False
            btc  = 0
            while not done:
                start = btc*btc_size
                end   = min((btc+1)*btc_size, lgt)

                agent.train_critic(start, end)

                btc += 1
                if (end == lgt): done = True
