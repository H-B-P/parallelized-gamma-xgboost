
import pandas as pd
import numpy as np
import xgboost as xgb

import random


x1Lookup = {1:1, 2:1, 3:3, 4:3}
x2Lookup = {1:1, 2:2, 3:3, 4:2}
x3Lookup = {1:2, 2:1, 3:2, 4:3}
x4Lookup = {1:4, 2:1, 3:1, 4:1}

x1List = []
x2List = []
x3List = []
x4List = []
yList = []

for i in range(100):
 x1 = random.choice([1,2,3,4])
 x2 = random.choice([1,2,3,4])
 x3 = random.choice([1,2,3,4])
 x4 = random.choice([1,2,3,4])
 y = x1Lookup[x1]*x2Lookup[x2]*x3Lookup[x3]*x4Lookup[x4] + random.choice([-0.2, 0, 0.2])
 x1List.append(x1)
 x2List.append(x2)
 x3List.append(x3)
 x4List.append(x4)
 yList.append(y)
 
dictForDf = {"x1":x1List,"x2":x2List, "x3":x3List, "x4":x4List, "y":yList}

df = pd.DataFrame(dictForDf)

print(df)

df.to_csv("dset.csv")