import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv("dset2.csv")

dtrain = xgb.DMatrix(df[["x1","x2","x3","x4"]], label=df["y"])

def gamma_grad(preds, acts):
 return (acts-preds)/preds

def gamma_hess(preds, acts):
 return -acts/preds

def gamma_obj(preds, dtrain):
 acts = dtrain.get_label()
 preds = np.exp(preds)
 grad = -gamma_grad(preds, acts)
 hess = -gamma_hess(preds, acts)
 return grad,hess

bst = xgb.train({"max_depth":1}, dtrain, 1000, obj = gamma_obj)

preds = np.exp(np.array(bst.predict(dtrain)))
acts = np.array(df["y"])

print(preds[:12])
print(acts[:12])

Lump = np.array([0]*len(df))

def gamma_grad_around_Lump(preds, acts): #Haskell forgive me for this parody of functional programming
 return (preds/((preds+Lump)**2))*(acts-(preds+Lump))

def gamma_hess_around_Lump(preds, acts):
 return (preds/((preds+Lump)**3))*(acts*Lump-Lump*preds-Lump*Lump-preds*acts)

def gamma_obj_around_Lump(preds, dtrain):
 acts = dtrain.get_label()
 preds = np.exp(preds)
 grad = -gamma_grad_around_Lump(preds,acts)
 hess = -gamma_hess_around_Lump(preds,acts)
 return grad, hess

bst = xgb.train({"max_depth":1}, dtrain, 1000, obj = gamma_obj_around_Lump)

preds = np.exp(np.array(bst.predict(dtrain)))
acts = np.array(df["y"])

print(preds[:12])
print(acts[:12])


BV = np.log(0.5*sum(df["y"])/len(df))

bstA = xgb.train({"max_depth":1, "base_score":BV}, dtrain, 0, obj = gamma_obj_around_Lump)
bstB = xgb.train({"max_depth":1, "base_score":BV}, dtrain, 0, obj = gamma_obj_around_Lump)


for i in range(30):
 Lump = np.array(np.exp(bstB.predict(dtrain)))
 bstA = xgb.train({"max_depth":1, "base_score":BV}, dtrain, i, obj = gamma_obj_around_Lump, xgb_model = bstA)
 Lump = np.array(np.exp(bstA.predict(dtrain)))
 bstB = xgb.train({"max_depth":1, "base_score":BV}, dtrain, i, obj = gamma_obj_around_Lump, xgb_model = bstB)

predsA = np.array(np.exp(bstA.predict(dtrain)))
predsB = np.array(np.exp(bstB.predict(dtrain)))
preds = predsA + predsB
acts = np.array(df["y"])

print(predsA[:12])
print(predsB[:12])
print(preds[:12])
print(acts[:12])

print(df[:12])