# Generic imports
import matplotlib.pyplot as plt

# Custom imports
from dragonfly.src.core.constants import *
from dragonfly.src.utils.buff     import *
from dragonfly.src.utils.counter  import *

###############################################
### Base srl
class base_srl():
    def __init__(self):
        pass

    def reset(self):
        pass

    def store_obs(self, obs):

        self.gbuff.store(["obs"], obs)
        self.counter += len(obs)

    def plot2Dencoded(self, obs):

        filename = 'compression2D'+str(self.counter)
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        #fig.suptitle(filename)

        ax.set_title('observations compressed in 2D')
        #ax.set_xlabel('')
        #ax.set_ylabel('')

        t = np.arange(len(obs))
        ax.scatter(obs[:,0],obs[:,1],c=t)
        
        ax.grid(True)
        ax.legend()
        
        fig.tight_layout()
        fig.savefig(filename+'.png')
