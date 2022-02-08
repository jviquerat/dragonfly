# Generic imports
import matplotlib.pyplot       as plt

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.titleweight'] = 'bold'

# Plot averaged fields
def plot_avg(data, filename):

    gen        = data[:,0]
    score_avg  = data[:,4]
    score_p    = data[:,5]
    score_m    = data[:,6]
    length_avg = data[:,10]
    length_p   = data[:,11]
    length_m   = data[:,12]

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle(filename)

    ax[0].set_title('score')
    ax[0].set_xlabel('episodes')
    ax[0].plot(score_avg,
               color='blue',
               label='avg')
    ax[0].fill_between(gen, score_p, score_m,
                       alpha=0.4,
                       color='blue',
                       label="+/- std")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_title('length')
    ax[1].set_xlabel('episodes')
    ax[1].plot(length_avg,
               color='red',
               label='avg', )
    ax[1].fill_between(gen, length_p, length_m,
                       alpha=0.4,
                       color='red',
                       label="+/- std")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(filename+'.png')
