import numpy as np
import matplotlib.pyplot as plt


def visualize_one_sample(sample, axes):
    '''
    Parameters
    ----------
    sample: np.ndarray
    axes: np.ndarray that contains matplotlib axes. The desired shape of axes is (5,).

    References
    ----------
    https://dacon.io/competitions/official/235646/codeshare/1706
    '''
    plt.style.use('fivethirtyeight')
    color_map = plt.cm.get_cmap('RdBu')
    color_map = color_map.reversed()

    for i, t in enumerate(['t-30', 't-20', 't-10', 't', 't+10']):
        if i < sample.shape[-1]:
            axes[i].imshow(sample[:, :, i], cmap=color_map)
        else:
            axes[i].imshow(np.zeros((120, 120)))

        axes[i].set_xlabel(f'{t}', fontsize=12)
        axes[i].set_xticks([])
        axes[i].set_yticks([])