# Generic imports
import numpy as np

###############################################
### Generic routine for advantage computation
### rwd        : reward array
### val        : value array
### nxt        : value array shifted by one timestep
### trm        : array of terminal values
### gamma      : discount factor
### gae_lambda : discount for GAE computation
### norm_adv   : whiten advantage array if true
### adv_clip   : clip   advantage array if true
def compute_adv(rwd, val, nxt, trm,
                gamma      = None,
                gae_lambda = None,
                norm_adv   = None,
                clip_adv   = None):

    # Handle arguments
    if (gamma      is None): gamma      = 0.99
    if (gae_lambda is None): gae_lambda = 0.99
    if (norm_adv   is None): norm_adv   = True
    if (adv_clip   is None): adv_clip   = True

    # Handle mask from termination signals
    msk = np.zeros(len(trm))
    for i in range(len(trm)):
        if (trm[i] == 0): msk[i] = 1.0
        if (trm[i] == 1): msk[i] = 0.0
        if (trm[i] == 2): msk[i] = 1.0

    # Compute deltas
    buff = zip(rwd, msk, nxt, val)
    dlt  = [r + gamma*m*nv - v for r, m, nv, v in buff]
    dlt  = np.stack(dlt)

    # Modify termination mask for GAE
    msk2 = np.zeros(len(trm))
    for i in range(len(trm)):
        if (trm[i] == 0): msk2[i] = 1.0
        if (trm[i] == 1): msk2[i] = 0.0
        if (trm[i] == 2): msk2[i] = 0.0

    # Compute advantages
    adv = dlt.copy()
    for t in reversed(range(len(adv)-1)):
        adv[t] += msk2[t]*gamma*gae_lambda*adv[t+1]

    # Compute targets
    tgt  = adv.copy()
    tgt += val

    # Normalize
    if norm_adv: adv = (adv-np.mean(adv))/(np.std(adv) + 1.0e-5)

    # Clip if required
    if clip_adv: adv = np.maximum(adv, 0.0)

    return tgt, adv
