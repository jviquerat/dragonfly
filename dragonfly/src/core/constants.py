########################
# Constants
########################

### Generic epsilon
eps = 1.0e-8

### Small epsilon for return normalization
ret_eps = 1.0e-8

### Small epsilon for PPO loss
ppo_eps = 1.0e-8

### Data for report
### freq_report is the nb of times the report will be
### dropped to file during a single run of training
### step_report is the time granularity of the content
### of the report itself, in terms of steps
freq_report = 10
step_report = 10
