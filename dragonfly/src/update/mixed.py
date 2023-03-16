###############################################
### Class for mixed update
class mixed():
    def __init__(self):
        self.reset()

    def reset(self):
        self.warmup = 3
        self.i      = 0

    # Update
    def update(self, agent, n_buff,
               size, btc_size, n_epochs):

        if (self.i > self.warmup):
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

        self.i += 1
