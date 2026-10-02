"""Train-only, persistent patch residual detection; not a glass classifier."""
import numpy as np


def patch_mse(prediction, target, patch_size=32):
    error = np.mean((prediction-target)**2, axis=0)
    h,w = error.shape
    # Include partial border patches without zero-padding their denominator.
    rows = np.arange(0,h,patch_size); cols = np.arange(0,w,patch_size)
    sums = np.add.reduceat(np.add.reduceat(error,rows,axis=0),cols,axis=1)
    areas = np.diff(np.r_[rows,h])[:,None] * np.diff(np.r_[cols,w])[None,:]
    return sums/areas, float(error.mean())


def persistent_candidates(history, ratio=2.0, floor=1e-4, persistence=.8,
                          max_improvement=.1, min_observations=4):
    if len(history) < min_observations:
        raise ValueError('Not enough independent late-training observations')
    maps = np.stack([v['mse'] for v in history])
    thresholds = np.array([max(v['global_mse']*ratio,floor) for v in history])
    high = maps > thresholds[:,None,None]
    frequency = high.mean(axis=0)
    first = maps[:max(1,len(maps)//2)].mean(axis=0)
    last = maps[-max(1,len(maps)//2):].mean(axis=0)
    improvement = (first-last)/np.maximum(first,1e-12)
    selected = (frequency >= persistence) & (improvement <= max_improvement) & high[-1]
    return selected, {'frequency':frequency,'relative_improvement':improvement,'last_mse':maps[-1]}


def expand_patches(mask, h, w, patch_size):
    return np.repeat(np.repeat(mask,patch_size,0),patch_size,1)[:h,:w]


def select_points(votes, scores, min_views=2, max_fraction=.2):
    """Cap candidate count, never force candidates when evidence is absent."""
    eligible = np.flatnonzero(votes >= min_views)
    limit = max(0,int(len(votes)*max_fraction))
    if len(eligible) > limit:
        eligible = eligible[np.argsort(-scores[eligible],kind='stable')[:limit]]
    result = np.zeros(len(votes), dtype=bool)
    result[eligible] = True
    return result
