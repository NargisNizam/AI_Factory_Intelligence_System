import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import random, os

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; IMG=DATA/"product_images"
IMG.mkdir(parents=True,exist_ok=True)
np.random.seed(42); random.seed(42)

n=1800
machine=np.random.choice(["PRESS-01","PRESS-02","CNC-01","CNC-02"],n)
temp=np.random.normal(72,8,n)
vib=np.random.normal(3.4,1.2,n)
pressure=np.random.normal(101,7,n)
rpm=np.random.normal(1450,130,n)
load=np.clip(np.random.normal(.72,.14,n),.2,.99)
defects=((temp>82)+(vib>5.2)+(load>.9)+ (pressure<88)).astype(int)
failure=((temp>84)&(vib>5.5) | (vib>6.3) | ((temp>80)&(load>.93))).astype(int)
df=pd.DataFrame({"machine_id":machine,"temperature":temp,"vibration":vib,"pressure":pressure,"rpm":rpm,"load":load,"defect":(defects>0).astype(int),"failure_next_24h":failure})
df.loc[np.random.choice(n,35,False),"pressure"]=np.nan
df=df.drop_duplicates()
df.to_csv(DATA/"machines.csv",index=False)

t=pd.date_range("2026-01-01",periods=3000,freq="10min")
base=72+5*np.sin(np.arange(len(t))/100)
s=pd.DataFrame({"timestamp":t,"temperature":base+np.random.normal(0,2,len(t)),
"vibration":3.2+.7*np.sin(np.arange(len(t))/70)+np.random.normal(0,.35,len(t)),
"pressure":101+4*np.sin(np.arange(len(t))/130)+np.random.normal(0,1.5,len(t)),
"rpm":1450+90*np.sin(np.arange(len(t))/90)+np.random.normal(0,35,len(t))})
s["temp_roll_12"]=s.temperature.rolling(12,min_periods=1).mean()
s["vib_roll_12"]=s.vibration.rolling(12,min_periods=1).mean()
s.to_csv(DATA/"sensors.csv",index=False)

notes=[
"Operator noticed elevated bearing vibration and unusual noise during the morning shift.",
"Routine inspection completed; cooling system clean and machine stable.",
"Temperature rising under high load. Supervisor requested inspection of lubrication.",
"Product surface showed a small scratch; tooling alignment checked.",
"Repeated pressure fluctuation reported; filter and regulator inspected.",
"Emergency maintenance performed after critical vibration alarm."
]
rows=[]
for i in range(120):
    rows.append({"machine_id":random.choice(["PRESS-01","PRESS-02","CNC-01","CNC-02"]),
                 "note":random.choice(notes),"urgency":random.choice(["low","medium","high"])})
pd.DataFrame(rows).to_csv(DATA/"maintenance_notes.csv",index=False)

# Synthetic product images: circles/rectangles represent acceptable vs defective surface patterns.
for i in range(240):
    defective=i%2==1
    im=Image.new("RGB",(96,96),"white"); d=ImageDraw.Draw(im)
    d.rectangle((12,12,84,84),outline="black",width=3)
    if defective:
        for _ in range(7):
            x=random.randint(20,75); y=random.randint(20,75)
            d.ellipse((x,y,x+5,y+5),fill="black")
    else:
        d.ellipse((35,35,61,61),outline="black",width=2)
    im.save(IMG/f'{"defect" if defective else "ok"}_{i:03d}.png')
print("Synthetic multimodal data generated.")
