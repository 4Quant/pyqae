"""A set of tools to piggy-back or enhance the scikit-learn tools"""
from sklearn.model_selection import train_test_split
import numpy as np
from hashlib import md5
good_hash = lambda x: int(md5(x).hexdigest(), 16)
uvals = lambda *args: np.unique(np.concatenate([x for x in args if x.shape[0] >0]))
def order_preserving_test_split(*in_data,
                                test_size,
                                random_state,
                                id_ext_func = lambda x: str(x).encode('ascii'),
                                id_idx = 0):
    """
    The challenge we want to solve here is having a consistent, reproducible
    train/test split as our dataset grows, so adding new datasets does not
    change
    :param in_data: the set of groups to split
    :param test_size: the fraction (very approximate) of the group
    :param random_state: offset for the hashes (helps to balance groups)
    :param id_ext_func: a function to extract the unique identifier for a list item
    :param id_idx: the input argument to determine the indexes on
    :return:
    >>> kwargs = dict(test_size = 0.5, random_state = 2017)
    >>> x1, y1 = order_preserving_test_split(['a', 'b'], **kwargs)
    >>> print(x1, y1)
    ['a'] ['b']
    >>> x1a, y1a = order_preserving_test_split(['b', 'a'], **kwargs)
    >>> print(x1a, y1a)
    ['a'] ['b']
    >>> x2, y2 = order_preserving_test_split(['a', 'b', 'c'], **kwargs)
    >>> x3, y3 = order_preserving_test_split(['b', 'c', 'd'],  **kwargs)
    >>> x4, y4 = order_preserving_test_split(['a', 'c', 'd'],  **kwargs)
    >>> train_set = uvals(x1,x1a,x2,x3,x4)
    >>> test_set = uvals(y1,y1a,y2,y3,y4)
    >>> print(train_set, test_set)
    ['a'] ['b' 'c' 'd']
    >>> [(x,'Contaminated' if x in train_set else 'Usuable') for x in test_set]
    [('b', 'Usuable'), ('c', 'Usuable'), ('d', 'Usuable')]
    >>> order_preserving_test_split(['a', 'b'], [1, 2], **kwargs)
    ([array(['a'],
          dtype='<U1'), array([1])], [array(['b'],
          dtype='<U1'), array([2])])
    """
    # look at the unique ids for each column
    idx_col = [id_ext_func(x) for x in in_data[id_idx]]
    group_div = 100
    np.random.seed(random_state)
    hash_offset = np.random.choice(range(group_div))

    hash_col = [good_hash(x)+hash_offset for x in idx_col]
    hash_in_test = np.array([(x % group_div)<int(test_size*group_div) for x
                             in  hash_col])
    train_out = [np.array(c_col)[hash_in_test==False] for c_col in in_data]
    test_out = [np.array(c_col)[hash_in_test] for c_col in in_data]
    if len(in_data)<2:
        # simplification that sklearn also does
        train_out = train_out[0]
        test_out = test_out[0]
    return train_out, test_out