#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from pandas import Series, DataFrame
from ast import literal_eval
from itertools import combinations
from datetime import datetime

comb_dict = {n: list(combinations(range(n), n-3)) for n in range(4, 15)}

def to_matrix(s):
    """Convert string of list of lists to np.array
    """
    res = np.array(literal_eval(s))
    return res.astype(np.float32)

def to_array(s):
    """Convert string of list of lists to np.array
    """
    return list(literal_eval(s))

def to_integer_kernel(a):
    n = len(a[0])
    A = matrix(ZZ, a).transpose()
    A_null = A.integer_kernel() # (n-3) x n
    return Matrix(A_null.basis())

def np_to_pluecker(a):
    n = len(a[0])
    B = to_integer_kernel(a)
    res = []
    for c in comb_dict[n]:
        minor = B[:, c]
        res.append(minor.det())
    return res 

def pluecker_csv(PATH):
    df = pd.read_csv(PATH, converters={"matrix":to_array}, header=None, names=["matrix", "pic"])
    df["pluecker"] = df["matrix"].apply(np_to_pluecker)
    pluecker_file = "pluecker_" + PATH
    df.to_csv(pluecker_file, index=False)
    
if __name__ == "__main__":
    pluecker_csv(PATH)


# In[ ]:




