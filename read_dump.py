#Incomplete, broken, not novel, do not use in current state

import numpy as np
import pandas as pd

treeshere = 100
treesdone = 0

f = open("dump.txt")

niceModel = {"base":0.5, "conts":{"x1":[], "x2":[],"x3":[],"x4":[]}}

def pred(df, model):
 df["preds"]=model["base"]
 for col in model["conts"]:
  df["mult"]=1
  for thing in model["conts"][col]
   df["mult"][df[col]>=thing[0]] = thing[1]
  df["preds"]=df["preds"]*df["mult"]
 return df["preds"]
  

while treesdone>treeshere:
 deadline = f.readline()
 specline = f.readline()
 yesline = f.readline()
 noline = f.readline()
 
 spec = specline.split("[")[1].split("]")[0]
 specfeat, specval = spec.split("<")
 specval=float(specval)
 
 if specval in [c for c in niceModel["conts"][specfeat]
 
 yesshift = np.exp(float(yesline.split("=")[1]))
 noshift = np.exp(float(noline.split("=")[1]))