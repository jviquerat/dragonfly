# Generic imports
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'

# Plot averaged fields
def plot_avg(data, filename):

    stp         = data[:,0]
    score_avg   = data[:,5]
    score_p     = data[:,6]
    score_m     = data[:,7]

    fig, ax = plt.subplots(1,1,figsize=(5,5))
    fig.suptitle(filename)

    ax.set_title('score')
    ax.set_xlabel('transitions')
    ax.plot(stp, score_avg,
            color='blue',
            label='avg')
    ax.fill_between(stp, score_p, score_m,
                    alpha=0.4,
                    color='blue',
                    label="+/- std")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename+'.png')
