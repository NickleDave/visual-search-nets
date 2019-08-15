"""functions for computing metrics: error, probabilities, etc."""
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import tensorflow_probability as tfp

z_score = norm.ppf


def compute_d_prime(y_true, y_pred):
    """computes d prime given y_true and y_pred

    adapted from <https://lindeloev.net/calculating-d-in-python-and-php/>
    """
    hits = np.logical_and(y_pred == 1, y_true == 1).sum()
    misses = np.logical_and(y_pred == 0, y_true == 1).sum()
    hit_rate = hits / (hits + misses)

    false_alarms = np.logical_and(y_pred == 1, y_true == 0).sum()
    correct_rejects = np.logical_and(y_pred == 0, y_true == 0).sum()
    false_alarm_rate = false_alarms / (false_alarms + correct_rejects)

    # standard correction to avoid d' value of infinity or minus infinity;
    # if either is 0 or 1, assume "true" value is somewhere between 0 (or 1)
    # and (1/2N) where N is the number of targets (or "lures", as appropriate)
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (false_alarms + correct_rejects)

    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    if false_alarm_rate == 1:
        false_alarm_rate = 1 - half_fa
    if false_alarm_rate == 0:
        false_alarm_rate = half_fa

    d_prime = z_score(hit_rate) - z_score(false_alarm_rate)
    return hit_rate.item(), false_alarm_rate.item(), d_prime.item()


def d_prime_tf(y_true, y_pred):
    """computes d prime given y_true and y_pred.

    Same as compute_d_prime function, but written as a Tensorflow computation graph,
    so it can be used as part of loss function (see searchnets.train).
    Adapted from <https://lindeloev.net/calculating-d-in-python-and-php/>.

    Parameters
    ----------
    y_true : Tensorflow.Tensor
    y_pred : Tensorflow.Tensor

    Returns
    -------
    d_prime : Tensorflow.Tensor
    """
    hits_vec = tf.cast(
        tf.math.logical_and(
            tf.math.equal(y_pred, tf.constant(1, tf.int32)),
            tf.math.equal(y_true, tf.constant(1, tf.int32))
                    ),dtype=tf.float32)
    hits = tf.math.reduce_sum(hits_vec)

    misses_vec = tf.cast(
        tf.math.logical_and(
            tf.math.equal(y_pred, tf.constant(0, tf.int32)),
            tf.math.equal(y_true, tf.constant(1, tf.int32))
        ), dtype=tf.float32)
    misses = tf.math.reduce_sum(misses_vec)

    hit_rate = hits / (hits + misses)

    false_alarms_vec = tf.cast(
        tf.math.logical_and(
            tf.math.equal(y_pred, tf.constant(1, tf.int32)),
            tf.math.equal(y_true, tf.constant(0, tf.int32))
                    ),dtype=tf.float32)
    false_alarms = tf.math.reduce_sum(false_alarms_vec)

    correct_rejects_vec = tf.cast(
        tf.math.logical_and(
            tf.math.equal(y_pred, tf.constant(0, tf.int32)),
            tf.math.equal(y_true, tf.constant(0, tf.int32))
                    ),dtype=tf.float32)
    correct_rejects = tf.math.reduce_sum(correct_rejects_vec)

    false_alarm_rate = false_alarms / (false_alarms + correct_rejects)

    # standard correction to avoid d' value of infinity or minus infinity;
    # if either is 0 or 1, assume "true" value is somewhere between 0 (or 1)
    # and (1/2N) where N is the number of targets (or "lures", as appropriate)
    half_hit = tf.constant(0.5, dtype=tf.float32) / (hits + misses)
    half_fa = tf.constant(0.5, dtype=tf.float32) / (false_alarms + correct_rejects)

    # if hit rate is 1, change to 1 - half hit
    hit_rate = tf.cond(
        pred=tf.math.equal(hit_rate, 1.0),
        true_fn=lambda: tf.constant(1.0, dtype=tf.float32) - half_hit,
        false_fn=lambda: hit_rate,
    )
    # if hit rate is 0, change to half hit
    hit_rate = tf.cond(
        pred=tf.math.equal(hit_rate, 0.0),
        true_fn=lambda: half_hit,
        false_fn=lambda: hit_rate,
    )

    false_alarm_rate = tf.cond(
        pred=tf.math.equal(false_alarm_rate, 1.0),
        true_fn=lambda: tf.constant(1.0, dtype=tf.float32) - half_fa,
        false_fn=lambda: false_alarm_rate,
    )
    false_alarm_rate = tf.cond(
        pred=tf.math.equal(false_alarm_rate, 0.0),
        true_fn=lambda: half_fa,
        false_fn=lambda: false_alarm_rate,
    )

    norm = tfp.distributions.Normal(loc=0, scale=1)
    z_score = norm.quantile  # rename quantile method (AKA ppf, inverse cdf) to z-score

    d_prime_op = z_score(hit_rate) - z_score(false_alarm_rate)
    return d_prime_op


def p_item_grid(char_grids, item_char='any', return_counts=False):
    """compute probability that item appears within each cell of grid

    Parameters
    ----------
    char_grids : list
        of lists or numpy.ndarrays, each representing a visual search stimulus
        as a grid of characters, where the character in each cell corresponds
        to some item type that can appear in the visual search stimulus.
        The searchstims library saves such a representation of each stimulus it
        generates in the .json metadata file that it saves.
    item_char : str
        Character that represents item for which probability should be computed.
        E.g., 't' is used for target and 'd' is used for distractor for most stimuli
        generated by searchstims.
        Default is 'any', which means 'count any character, regardless of what it is'.
    return_counts : bool
        if True, return array with count of number of occurrences of item_char in
        each cell. Default is False.

    Returns
    -------
    p : numpy.ndarray
        of same shape as char_grids, where value of each element is probability
        that item_char occurs in the corresponding cell in char_grids
    counts : numpy.ndarray
        counts used to compute p (by dividing by np.sum(counts)).
        Only returned if return_counts is True
    """
    char_grids = [np.asarray(g) for g in char_grids]

    grid_shape = [g.shape for g in char_grids]
    grid_shape = set(grid_shape)
    if len(grid_shape) == 1:
        grid_shape = grid_shape.pop()
    else:
        raise ValueError(
            'found more than one shape for visual search stimuli grids: '
            f'{grid_shape}'
        )

    counts = np.zeros(grid_shape)
    for g in char_grids:
        # increment by 1 the cell where the item type is found
        # using indices returned by np.nonzero
        if item_char == 'any':
            counts[np.nonzero(g != '')] += 1  # count any non-zero cell
        else:
            counts[np.nonzero(g == item_char)] += 1
    p = counts / np.sum(counts)
    if return_counts:
        return p, counts
    else:
        return p


def acc_grid(stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta, return_counts=False):
    """compute accuracy for each cell of grid in visual search stimulus

    Parameters
    ----------
    stim_fnames : list
    y_true : numpy.ndarray
    y_pred : numpy.ndarray
    char_grids : list
        of lists or numpy.ndarrays, each representing a visual search stimulus
        as a grid of characters, where the character in each cell corresponds
        to some item type that can appear in the visual search stimulus.
        The searchstims library saves such a representation of each stimulus it
        generates in the .json metadata file that it saves.
    stim_fnames_meta : list
    return_counts : bool
        if True, return array with count of number of occurrences of item_char in
        each cell. Default is False.

    Returns
    -------
    acc : numpy.ndarray
        of same shape as char_grids, where value of each element is probability
        that item_char occurs in the corresponding cell in char_grids
    counts : numpy.ndarray
        counts used to compute p (by dividing by np.sum(counts)).
        Only returned if return_counts is True
    """
    # convert to numpy.ndarray so we can use np.nonzero, check for multiple occurrences (not that we want to find them)
    stim_fnames = np.asarray(stim_fnames)
    stim_fnames_meta = np.asarray(stim_fnames_meta)

    char_grids = [np.asarray(g) for g in char_grids]

    grid_shape = [g.shape for g in char_grids]
    grid_shape = set(grid_shape)
    if len(grid_shape) == 1:
        grid_shape = grid_shape.pop()
    else:
        raise ValueError(
            'found more than one shape for visual search stimuli grids: '
            f'{grid_shape}'
        )

    correct_counts = np.zeros(grid_shape)
    trial_counts = np.zeros(grid_shape)
    for g, g_fname in zip(char_grids, stim_fnames_meta):
        # get indices in character grid that are not empty
        row_inds, col_inds = np.nonzero(g != '')

        # find index of filename that corresponds to grid in
        # list of filenames that corresponds to y_true and y_pred
        ind = np.nonzero(stim_fnames == g_fname)[0]
        if ind.shape[0] != 1:
            raise ValueError(
                f'Did not find only one index for {g_fname} in list of stimulus filenames made by searchnets.data.\n'
                f'Indices found were: {ind}'
            )
        ind = ind[0]

        is_correct = (y_true[ind] == y_pred[ind])
        if is_correct:
            for row_ind, col_ind in zip(row_inds, col_inds):
                correct_counts[row_ind, col_ind] += 1
        for row_ind, col_ind in zip(row_inds, col_inds):
            trial_counts[row_ind, col_ind] += 1

    acc = correct_counts / trial_counts
    if return_counts:
        return acc, correct_counts, trial_counts
    else:
        return acc


def err_grid(stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta, return_counts=False):
    if return_counts:
        acc, correct_counts, trial_counts = acc_grid(
            stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta, return_counts
        )
    else:
        acc = acc_grid(stim_fnames, y_true, y_pred, char_grids, stim_fnames_meta, return_counts)

    err = 1 - acc

    if return_counts:
        return err, correct_counts, trial_counts
    else:
        return err
