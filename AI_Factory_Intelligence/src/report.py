from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path

def make_report(path, machine, predictive, vision, nlp, recommendation, human_decision, rag, twin):
    doc=SimpleDocTemplate(str(path),pagesize=A4); styles=getSampleStyleSheet(); story=[]
    story += [Paragraph("AI Factory Incident & Decision Report",styles["Title"]),
              Paragraph("Autonomous Manufacturing Intelligence & Digital Twin",styles["Heading2"]),Spacer(1,12)]
    story.append(Paragraph(f"<b>Machine:</b> {machine}",styles["BodyText"]))
    story.append(Paragraph(f"<b>Predictive risk:</b> {predictive['failure_probability']:.1%} — {predictive['risk']}",styles["BodyText"]))
    story.append(Paragraph(f"<b>Vision:</b> {vision['defect']} ({vision['confidence']:.1%})",styles["BodyText"]))
    story.append(Paragraph(f"<b>NLP urgency:</b> {nlp['urgency']} | Symptoms: {', '.join(nlp['entities']) or 'none'}",styles["BodyText"]))
    story.append(Spacer(1,10))
    story.append(Paragraph("RAG Evidence",styles["Heading2"]))
    for r in rag: story.append(Paragraph(f"{r['source']}: {'; '.join(r['evidence'])}",styles["BodyText"]))
    story.append(Spacer(1,10))
    story.append(Paragraph(f"<b>AI Recommendation:</b> {recommendation}",styles["Heading2"]))
    story.append(Paragraph(f"<b>Human Supervisor Decision:</b> {human_decision}",styles["BodyText"]))
    story.append(Spacer(1,10))
    story.append(Paragraph("Digital Twin What-If Results",styles["Heading2"]))
    data=[["Scenario","Production","Risk","Downtime","Relative cost"]]
    for _,r in twin.iterrows(): data.append([r["scenario"],str(r["expected_production"]),str(r["risk"]),str(r["downtime_hours"]),str(r["relative_cost"])])
    t=Table(data); t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])); story.append(t)
    story.append(Spacer(1,12))
    story.append(Paragraph("Safety: This report is decision support. Final operational action requires qualified human approval.",styles["BodyText"]))
    doc.build(story)
