#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from pandas import Series, DataFrame
from ast import literal_eval
from itertools import combinations
import os

comb_dict = {n: list(combinations(range(n), n-3)) for n in range(4, 15)}

def to_integer_kernel(a):
    n = len(a[0])
    A = matrix(ZZ, a).transpose()
    A_null = A.integer_kernel() # (n-3) x n
    return Matrix(A_null.basis())

def array_to_pluecker(a):
    n = len(a[0])
    B = to_integer_kernel(a)
    res = []
    for c in comb_dict[n]:
        minor = B[:, c]
        res.append(minor.det())
    return res

def pluecker_csv(input_file):
    input_head, input_tail = os.path.split(input_file)
    output_file = input_head + "/pluecker_" + input_tail
    df = pd.read_csv(input_file, converters={"matrix":literal_eval})
    df["pluecker"] = df["matrix"].apply(array_to_pluecker)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    for input_file in input_file_paths:
        pluecker_csv(input_file)


# In[ ]:
