import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent/"src"))
import streamlit as st, pandas as pd, numpy as np
from pipeline import predictive_agent, vision_agent, nlp_agent, rag_agent, planning_agent, digital_twin
from report import make_report

st.set_page_config(page_title="AI Factory Command Center",layout="wide")
st.title("🏭 AI Factory Intelligence Command Center")
st.caption("Autonomous Manufacturing Intelligence & Digital Twin — Batch-12 | Decision support, not autonomous authority")

if not (Path("models/failure_model.joblib").exists()):
    st.warning("Models are not trained yet. Run: `python src/generate_data.py` then `python src/train_models.py`.")
    st.stop()

left,right=st.columns(2)
with left:
    st.subheader("1. Machine & Sensor Input")
    machine=st.selectbox("Machine",["PRESS-01","PRESS-02","CNC-01","CNC-02"])
    temperature=st.number_input("Temperature (°C)",50.0,110.0,86.0)
    vibration=st.number_input("Vibration (mm/s)",0.0,12.0,5.8)
    pressure=st.number_input("Pressure",60.0,130.0,98.0)
    rpm=st.number_input("RPM",500.0,2500.0,1480.0)
    load=st.slider("Machine load",0.1,1.0,0.92)
    note=st.text_area("Maintenance / incident note","Operator noticed elevated bearing vibration and rising temperature under high load.")
with right:
    st.subheader("2. Product Image")
    image=st.file_uploader("Upload product image",type=["png","jpg","jpeg"])
    if image: st.image(image,width=260)

values={"temperature":temperature,"vibration":vibration,"pressure":pressure,"rpm":rpm,"load":load}
if st.button("RUN FULL FACTORY INTELLIGENCE PIPELINE",type="primary"):
    vision=vision_agent(image); pred=predictive_agent(values); nlp=nlp_agent(note)
    query=f"{note} temperature {temperature} vibration {vibration} maintenance safety"
    rag=rag_agent(query); plan=planning_agent(vision,pred,nlp,rag); twin=digital_twin(values)
    st.session_state.update({"vision":vision,"pred":pred,"nlp":nlp,"rag":rag,"plan":plan,"twin":twin,"machine":machine})

if "pred" in st.session_state:
    pred=st.session_state["pred"]; vision=st.session_state["vision"]; nlp=st.session_state["nlp"]
    rag=st.session_state["rag"]; plan=st.session_state["plan"]; twin=st.session_state["twin"]
    st.divider(); st.subheader("3. AI Evidence & Explainability")
    c1,c2,c3=st.columns(3)
    c1.metric("Failure probability",f"{pred['failure_probability']:.1%}")
    c2.metric("Visual result",vision["defect"])
    c3.metric("NLP urgency",nlp["urgency"].upper())
    st.write("**XAI — influential signals:** temperature, vibration and load are evaluated by the predictive model. "
             f"Current values: {pred['signals']}.")
    st.write("**Vision evidence:**",vision["evidence"])
    st.subheader("4. RAG Knowledge Evidence")
    for r in rag:
        st.info(f"Source: {r['source']} | " + " ".join(r["evidence"]))
    st.subheader("5. Multi-Agent Decision")
    st.write("Vision Agent →",vision)
    st.write("Predictive Maintenance Agent →",pred)
    st.write("Knowledge/NLP Agents →",{"nlp":nlp,"rag":rag})
    st.success("Planning/Decision Agent recommendation: "+plan["recommendation"])
    st.write("Reasons:",", ".join(plan["reasons"]))
    st.subheader("6. Digital Twin — What-If Simulation")
    st.dataframe(twin,use_container_width=True)
    best=twin.sort_values(["risk","relative_cost"]).iloc[0]
    st.caption(f"Simulation insight: {best['scenario']} gives the lowest combined risk/cost among these simplified scenarios.")
    st.subheader("7. Human-in-the-Loop")
    decision=st.radio("Supervisor decision",["APPROVE","REJECT","MODIFY"],horizontal=True)
    reason=st.text_input("Supervisor reason / modification","")
    if st.button("Generate Final PDF Report"):
        out=Path("reports/factory_incident_report.pdf")
        make_report(out,st.session_state["machine"],pred,vision,nlp,plan["recommendation"],f"{decision}: {reason}",rag,twin)
        with open(out,"rb") as f: st.download_button("Download PDF report",f,file_name="AI_Factory_Incident_Report.pdf",mime="application/pdf")
    st.subheader("8. MLOps")
    st.write("MLflow experiments: Logistic Regression baseline, Random Forest, Gradient Boosting. Best selected model is saved as `models/failure_model.joblib`.")
