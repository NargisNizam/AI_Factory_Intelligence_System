import os, sys, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"; MODELS=ROOT/"models"; MODELS.mkdir(exist_ok=True)
df=pd.read_csv(DATA/"machines.csv")
X=df[["temperature","vibration","pressure","rpm","load"]]; y=df.failure_next_24h
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models=[
("logistic_baseline",LogisticRegression(max_iter=1000)),
("random_forest",RandomForestClassifier(n_estimators=180,max_depth=8,random_state=42)),
("gradient_boosting",GradientBoostingClassifier(random_state=42))
]
best=None
try:
    import mlflow
    mlflow.set_tracking_uri("file:"+str(ROOT/"mlruns"))
except Exception:
    mlflow=None

results=[]
for name,model in models:
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",model)])
    if mlflow: mlflow.set_experiment("AI_Factory")
    runctx=mlflow.start_run(run_name=name) if mlflow else None
    pipe.fit(Xtr,ytr); p=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
    m={"accuracy":accuracy_score(yte,p),"precision":precision_score(yte,p,zero_division=0),
       "recall":recall_score(yte,p,zero_division=0),"f1":f1_score(yte,p,zero_division=0),
       "roc_auc":roc_auc_score(yte,prob)}
    results.append((name,m))
    if mlflow:
        mlflow.log_params({"model":name}); mlflow.log_metrics(m)
        mlflow.end_run()
    if best is None or m["f1"]>best[1]["f1"]: best=(name,m,pipe)

joblib.dump(best[2],MODELS/"failure_model.joblib")
pd.DataFrame([{"experiment":n,**m} for n,m in results]).to_csv(MODELS/"experiment_results.csv",index=False)

# Deep-learning style predictive model using sklearn MLP (fast, CPU-friendly).
mlp=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",MLPClassifier(hidden_layer_sizes=(32,16),max_iter=300,random_state=42,early_stopping=True))])
mlp.fit(Xtr,ytr); joblib.dump(mlp,MODELS/"deep_predictive_model.joblib")

# Simple image model metadata; Streamlit can use visual rules for the synthetic image demo.
with open(MODELS/"image_model.txt","w") as f: f.write("Synthetic CNN-equivalent demo: defect vs OK based on image anomaly pattern. Replace with a real CNN/transfer-learning model for production data.\n")
print("Best model:",best[0],best[1])
print(pd.DataFrame([{"experiment":n,**m} for n,m in results]))
