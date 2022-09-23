########################
# Constants
########################

### Small epsilon for return normalization
ret_eps = 1.0e-8

### Small epsilon for PPO loss
ppo_eps = 1.0e-8

### Smoothing horizon for score
n_smooth = 1000

### Max value for observations
def_obs_max = 1.0e8

### Frequency at which report is written
### This is the nb of times the report will be
### dropped during a single run of training
freq_report = 10
