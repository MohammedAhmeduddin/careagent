"""
Generate synthetic CMS Medicare provider data with the exact
same column structure as the real CMS dataset.
Used for development and demo. 10,000 providers, ~50,000 procedure rows.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

N_PROVIDERS = 10_000
ROWS_PER_PROVIDER = 5  # avg procedures per provider

SPECIALTIES = [
    "Internal Medicine", "Family Practice", "Cardiology",
    "Orthopedic Surgery", "Dermatology", "Psychiatry",
    "Ophthalmology", "General Surgery", "Neurology",
    "Gastroenterology", "Pulmonary Disease", "Nephrology",
    "Oncology", "Anesthesiology", "Radiology",
    "Emergency Medicine", "Obstetrics & Gynecology", "Urology",
    "Rheumatology", "Endocrinology",
]

HCPCS = [
    ("99213", "Office visit established patient moderate"),
    ("99214", "Office visit established patient moderate-high"),
    ("99203", "Office visit new patient moderate"),
    ("99232", "Subsequent hospital care"),
    ("93000", "Electrocardiogram routine"),
    ("71046", "Chest X-ray 2 views"),
    ("80053", "Comprehensive metabolic panel"),
    ("85025", "Complete blood count"),
    ("36415", "Venipuncture"),
    ("90837", "Psychotherapy 60 minutes"),
    ("27447", "Total knee replacement"),
    ("33533", "Coronary artery bypass"),
    ("43239", "Upper GI endoscopy biopsy"),
    ("66984", "Cataract surgery"),
    ("70553", "MRI brain with contrast"),
]

STATES = [
    "CA","TX","FL","NY","PA","IL","OH","GA","NC","MI",
    "NJ","VA","WA","AZ","MA","TN","IN","MO","MD","WI",
]

FIRST_NAMES = ["James","Mary","John","Patricia","Robert","Jennifer","Michael",
               "Linda","William","Barbara","David","Susan","Richard","Jessica"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
               "Davis","Wilson","Anderson","Taylor","Thomas","Jackson","White"]
CREDENTIALS = ["MD","DO","MD","MD","DO","NP","PA","MD","MD","DO"]

rows = []
npis = [str(random.randint(1000000000, 9999999999)) for _ in range(N_PROVIDERS)]

for npi in npis:
    specialty   = random.choice(SPECIALTIES)
    state       = random.choice(STATES)
    entity_type = random.choices(["I","O"], weights=[0.85, 0.15])[0]
    gender      = random.choices(["M","F",""], weights=[0.55, 0.40, 0.05])[0]
    base_pay    = np.random.lognormal(mean=5.0, sigma=0.6)  # realistic payment dist

    # Inject ~3% anomalies — high cost, low volume
    is_anomaly = random.random() < 0.03
    if is_anomaly:
        base_pay *= random.uniform(3.5, 6.0)

    n_procs = random.randint(1, 10)
    proc_sample = random.sample(HCPCS, min(n_procs, len(HCPCS)))

    for hcpcs_code, hcpcs_desc in proc_sample:
        svc_count  = max(1, int(np.random.lognormal(4.5, 1.0)))
        bene_count = max(11, int(svc_count * random.uniform(0.4, 0.9)))
        pay_amt    = base_pay * random.uniform(0.8, 1.2)
        charge_amt = pay_amt  * random.uniform(1.8, 3.5)
        allowed    = pay_amt  * random.uniform(1.05, 1.25)

        # Suppress ~8% of beneficiary counts (CMS privacy rule)
        if random.random() < 0.08:
            bene_count = ""

        rows.append({
            "Rndrng_NPI":                     npi,
            "Rndrng_Prvdr_Last_Org_Name":     random.choice(LAST_NAMES) if entity_type=="I" else f"{random.choice(LAST_NAMES)} Medical Group",
            "Rndrng_Prvdr_First_Name":        random.choice(FIRST_NAMES) if entity_type=="I" else "",
            "Rndrng_Prvdr_MI":                random.choice(["A","B","C",""]),
            "Rndrng_Prvdr_Crdntls":          random.choice(CREDENTIALS) if entity_type=="I" else "",
            "Rndrng_Prvdr_Gndr":             gender if entity_type=="I" else "",
            "Rndrng_Prvdr_Ent_Cd":           entity_type,
            "Rndrng_Prvdr_St1":              f"{random.randint(1,9999)} Main St",
            "Rndrng_Prvdr_City":             "Springfield",
            "Rndrng_Prvdr_State_Abrvtn":     state,
            "Rndrng_Prvdr_Zip5":             str(random.randint(10000,99999)),
            "Rndrng_Prvdr_RUCA":             str(random.randint(1,10)),
            "Rndrng_Prvdr_Cntry":            "US",
            "Rndrng_Prvdr_Type":             specialty,
            "Rndrng_Prvdr_Mdcr_Prtcptn_Ind": random.choices(["Y","N"], weights=[0.95,0.05])[0],
            "HCPCS_Cd":                       hcpcs_code,
            "HCPCS_Desc":                     hcpcs_desc,
            "HCPCS_Drug_Ind":                 "N",
            "Place_Of_Srvc":                  random.choice(["F","O"]),
            "Tot_Benes":                      bene_count,
            "Tot_Srvcs":                      round(svc_count, 1),
            "Tot_Bene_Day_Srvcs":            round(svc_count * 1.1, 1),
            "Avg_Mdcr_Alowd_Amt":            round(allowed, 2),
            "Avg_Sbmtd_Chrg":                round(charge_amt, 2),
            "Avg_Mdcr_Pymt_Amt":             round(pay_amt, 2),
            "Avg_Mdcr_Stdzd_Amt":            round(pay_amt * 0.95, 2),
        })

df = pd.DataFrame(rows)
out = Path("data/cms_provider_2022.csv")
df.to_csv(out, index=False)

print(f"Generated {len(df):,} rows")
print(f"Unique providers: {df['Rndrng_NPI'].nunique():,}")
print(f"File size: {out.stat().st_size / 1024 / 1024:.1f} MB")
print(f"Columns: {list(df.columns)}")
