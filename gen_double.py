
import pandas as pd
import numpy as np
import xgboost as xgb

import random


x1ALookup = {1:1, 2:1, 3:3, 4:3}
x2ALookup = {1:1, 2:2, 3:3, 4:2}
x3ALookup = {1:2, 2:1, 3:2, 4:3}
x4ALookup = {1:4, 2:1, 3:1, 4:1}

x1BLookup = {1:1, 2:1, 3:2, 4:3}
x2BLookup = {1:1, 2:2, 3:3, 4:4}
x3BLookup = {1:2, 2:1, 3:2, 4:1}
x4BLookup = {1:1, 2:1, 3:1, 4:3}

x1List = []
x2List = []
x3List = []
x4List = []
aList = []
bList = []
yList = []

for i in range(1000):
 x1 = random.choice([1,2,3,4])
 x2 = random.choice([1,2,3,4])
 x3 = random.choice([1,2,3,4])
 x4 = random.choice([1,2,3,4])
 a = x1ALookup[x1]*x2ALookup[x2]*x3ALookup[x3]*x4ALookup[x4]
 b = x1BLookup[x1]*x2BLookup[x2]*x3BLookup[x3]*x4BLookup[x4]
 y = a + b + random.choice([-0.2, 0, 0.2])
 x1List.append(x1)
 x2List.append(x2)
 x3List.append(x3)
 x4List.append(x4)
 aList.append(a)
 bList.append(b)
 yList.append(y)
 
dictForDf = {"x1":x1List,"x2":x2List, "x3":x3List, "x4":x4List,"a":aList, "b":bList, "y":yList}

df = pd.DataFrame(dictForDf)

print(df)

df.to_csv("dset2.csv")