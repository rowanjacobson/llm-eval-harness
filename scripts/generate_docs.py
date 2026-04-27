import json
import random
from datetime import datetime, timedelta

departments = ["Security", "Engineering", "Product", "HR", "Finance", "Support"]
source_types = ["policy", "handbook", "memo", "roadmap", "runbook", "guideline"]

incident_roles = [
    "incident commander",
    "primary on-call engineer",
    "backup on-call engineer",
    "engineering manager",
    "release manager"
]

topics = [
    "incident response",
    "on-call procedures",
    "database access",
    "travel expenses",
    "remote work",
    "support SLA",
    "release process",
    "security approvals",
    "audit logging",
    "roadmap planning"
]

def random_date():
    start = datetime(2025, 1, 1)
    return (start + timedelta(days=random.randint(0, 500))).strftime("%Y-%m-%d")

def generate_text(doc_id):
    topic = random.choice(topics)

    if topic == "incident response":
        return f"""
Sev1 incidents must be escalated to the {random.choice(incident_roles)} within {random.choice([10,15,20])} minutes.
Updates must be posted every {random.choice([15,30,45])} minutes during active incidents.
Sev2 incidents must be escalated within {random.choice([30,60])} minutes.
"""

    elif topic == "on-call procedures":
        return f"""
The primary on-call engineer is responsible for initial triage.
If mitigation fails after {random.choice([1,2,3])} attempts, the backup on-call engineer must be paged.
Escalation to the engineering manager occurs after sustained failure.
"""

    elif topic == "database access":
        return f"""
Production database access requires approval from the Security team.
Access is granted for a maximum of {random.choice([3,5,7,10])} days.
Contractors require additional approval from a director-level employee.
"""

    elif topic == "travel expenses":
        return f"""
Hotel costs are capped at £{random.choice([150,180,200])} per night in London.
Flights above £{random.choice([400,500,600])} require manager approval.
Receipts are required for expenses above £{random.choice([20,25,50])}.
"""

    elif topic == "remote work":
        return f"""
Employees may work abroad for up to {random.choice([15,20,25])} days per year.
Work must be conducted from approved countries only.
Tax review is required after {random.choice([7,10,14])} consecutive days abroad.
"""

    elif topic == "support SLA":
        return f"""
Enterprise support tickets require a first response within {random.choice([1,2])} hour.
Standard tickets require a response within {random.choice([6,8,12])} business hours.
P1 issues must be escalated to the incident response process.
"""

    elif topic == "release process":
        return f"""
Code freeze begins at {random.choice(['Friday 17:00 UTC', 'Thursday 18:00 UTC'])} before major releases.
Hotfixes require approval from the release manager.
Rollback decisions during incidents are owned by the incident commander.
"""

    elif topic == "security approvals":
        return f"""
All production changes require approval from the Security team.
High-risk changes require additional review by a senior engineer.
Audit logs must be retained for {random.choice([30,60,90])} days.
"""

    elif topic == "audit logging":
        return f"""
Audit logs must capture all administrative actions.
Logs are stored for {random.choice([30,60,90])} days.
Access to logs is restricted to authorized personnel only.
"""

    elif topic == "roadmap planning":
        return f"""
This document outlines planned initiatives for the upcoming quarter.
Features under consideration include analytics improvements and authentication upgrades.
Final prioritization is subject to budget and staffing constraints.
"""

    return "General internal documentation."

documents = []

for i in range(1000):
    doc = {
        "doc_id": f"doc_{i:03}",
        "title": f"{random.choice(topics).title()} Document {i}",
        "department": random.choice(departments),
        "source_type": random.choice(source_types),
        "last_updated": random_date(),
        "text": generate_text(i).strip()
    }
    documents.append(doc)

with open("documents.json", "w") as f:
    json.dump(documents, f, indent=2)

print("Generated 1000 documents in documents.json")