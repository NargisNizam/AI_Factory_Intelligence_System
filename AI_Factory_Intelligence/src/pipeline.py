from pathlib import Path
import pandas as pd, numpy as np, joblib, re
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def predict_machine(values):
    model=joblib.load(ROOT/"models/failure_model.joblib")
    X=pd.DataFrame([values])
    prob=float(model.predict_proba(X)[0,1])
    label="HIGH RISK" if prob>=.5 else "LOW/MODERATE RISK"
    return label,prob

def vision_agent(image):
    if image is None: return {"defect":"not analyzed","confidence":0.0,"evidence":"No image uploaded."}
    im=Image.open(image).convert("L").resize((96,96)); a=np.array(im)
    # Dark-pixel density inside the product area is a transparent synthetic proxy.
    density=float((a[15:82,15:82]<80).mean())
    defect=density>.055
    conf=min(.99,max(.55,abs(density-.03)*8+.55))
    return {"defect":"surface defect" if defect else "OK","confidence":conf,
            "evidence":f"Visual anomaly density={density:.3f} in the inspected product region."}

def predictive_agent(values):
    label,prob=predict_machine(values)
    return {"risk":label,"failure_probability":prob,
            "signals":{"temperature":values["temperature"],"vibration":values["vibration"],
                       "pressure":values["pressure"],"rpm":values["rpm"],"load":values["load"]}}

def nlp_agent(note):
    txt=(note or "").lower()
    urgent=any(k in txt for k in ["emergency","critical","repeated","elevated","rising"])
    terms=[k for k in ["vibration","temperature","pressure","bearing","cooling","lubrication","alignment"] if k in txt]
    return {"urgency":"high" if urgent else "medium","entities":terms,"summary":"Maintenance note analyzed for urgency and machine symptoms."}

def rag_agent(query):
    docs=[]
    for p in (ROOT/"knowledge_base").glob("*.txt"):
        text=p.read_text(encoding="utf-8")
        score=sum(1 for w in re.findall(r"[a-zA-Z]+",query.lower()) if w in text.lower())
        docs.append((score,p.name,text))
    docs.sort(reverse=True)
    best=docs[:2]
    snippets=[]
    for score,name,text in best:
        lines=[x.strip() for x in text.splitlines() if x.strip()]
        hits=[x for x in lines if any(w in x.lower() for w in re.findall(r"[a-zA-Z]{4,}",query.lower()))]
        snippets.append({"source":name,"evidence":(hits[:3] or lines[:3]),"score":score})
    return snippets

def planning_agent(vision,predictive,nlp,rag):
    p=predictive["failure_probability"]
    critical=p>=.75 or predictive["signals"]["temperature"]>=92 or predictive["signals"]["vibration"]>=6.3
    action="STOP AND INSPECT" if critical else ("REDUCE LOAD + SCHEDULE INSPECTION" if p>=.5 or predictive["signals"]["temperature"]>=85 else "CONTINUE WITH MONITORING")
    why=[]
    if predictive["signals"]["temperature"]>=85: why.append("high temperature")
    if predictive["signals"]["vibration"]>=5.5: why.append("elevated vibration")
    if vision["defect"]!="OK": why.append("product visual defect")
    if nlp["urgency"]=="high": why.append("urgent maintenance note")
    return {"recommendation":action,"reasons":why or ["risk remains within monitored range"],"human_approval_required":True}

def digital_twin(values):
    temp=values["temperature"]; vib=values["vibration"]; load=values["load"]
    risk=min(1,max(0,(temp-65)/35*.55+(vib-2)/7*.45))
    scenarios=[]
    for name,load_factor in [("Continue operation",1.0),("Reduce load",.75),("Stop for maintenance",0.0)]:
        downtime=2 if load_factor==.75 else (0 if load_factor==1 else 6)
        prod=max(0,100*load*load_factor*(1-risk*.55))
        expected_risk=max(0,risk*(.45 if load_factor==.75 else (.05 if load_factor==0 else 1.0)))
        cost=round(downtime*120+max(0,100-prod)*8,2)
        scenarios.append({"scenario":name,"expected_production":round(prod,1),"risk":round(expected_risk,2),"downtime_hours":downtime,"relative_cost":cost})
    return pd.DataFrame(scenarios)
