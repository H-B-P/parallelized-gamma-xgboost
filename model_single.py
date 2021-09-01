import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv("dset.csv")

dtrain = xgb.DMatrix(df[["x1","x2","x3","x4"]], label=df["y"])

bst = xgb.train({"max_depth":1, 'objective':'reg:gamma'}, dtrain, 100)

preds = np.array(bst.predict(dtrain))
acts = np.array(df["y"])

print(preds)

def gamma_grad(preds, acts):
 return (preds-acts)/preds

def gamma_hess(preds, acts):
 return acts/preds

def gamma_obj(preds, dtrain):
 acts = dtrain.get_label()
 preds = np.exp(preds)
 grad = gamma_grad(preds, acts)
 hess = gamma_hess(preds, acts)
 return grad,hess

bst = xgb.train({"max_depth":1}, dtrain, 100, obj = gamma_obj)

preds = np.exp(np.array(bst.predict(dtrain)))
acts = np.array(df["y"])

print(preds)

bst.dump_model('dump.txt')