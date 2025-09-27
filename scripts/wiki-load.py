import json
import re
import time
from typing import List
from tqdm import tqdm

import wikipediaapi

# --- NLTK sentence tokenizer (handle both punkt variants) ---
import nltk
try:
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize

# ----------------------------
# 150 curated Wikipedia titles
# ----------------------------
TITLES: List[str] = [
    # Cardiovascular (20)
    "Hypertension","Coronary artery disease","Myocardial infarction","Heart failure",
    "Atrial fibrillation","Aortic dissection","Deep vein thrombosis","Pulmonary embolism",
    "Stroke","Transient ischemic attack","Aortic stenosis","Mitral regurgitation",
    "Endocarditis","Pericarditis","Cardiomyopathy","Peripheral artery disease",
    "Hyperlipidemia","Statin","Beta blocker","ACE inhibitor",

    # Endocrine & Metabolic (20)
    "Diabetes mellitus","Diabetic ketoacidosis","Hyperosmolar hyperglycemic state",
    "Metabolic syndrome","Obesity","Hypothyroidism","Hyperthyroidism",
    "Cushing's syndrome","Addison's disease","Polycystic ovary syndrome",
    "Osteoporosis","Vitamin D deficiency","Hyperparathyroidism","Hypoparathyroidism",
    "Insulin","Metformin","Levothyroxine","Liraglutide","SGLT2 inhibitor","HbA1c",

    # Respiratory (12)
    "Asthma","Chronic obstructive pulmonary disease","Pneumonia","Tuberculosis",
    "Pulmonary fibrosis","Sarcoidosis","Obstructive sleep apnea","Acute respiratory distress syndrome",
    "Bronchiectasis","Cystic fibrosis","Pleural effusion","Pneumothorax",

    # Infectious diseases (20)
    "Sepsis","Meningitis","Encephalitis","Urinary tract infection","HIV/AIDS","Hepatitis B",
    "Hepatitis C","Influenza","COVID-19","Malaria","Dengue fever","Typhoid fever",
    "Tetanus","Rabies","Syphilis","Gonorrhea","Chlamydia infection","Herpes simplex",
    "Varicella zoster","Clostridioides difficile infection",

    # Gastroenterology & Hepatology (14)
    "Gastroesophageal reflux disease","Peptic ulcer disease","Inflammatory bowel disease",
    "Crohn's disease","Ulcerative colitis","Irritable bowel syndrome","Celiac disease",
    "Pancreatitis","Cirrhosis","Non-alcoholic fatty liver disease","Hepatocellular carcinoma",
    "Upper gastrointestinal bleeding","Lower gastrointestinal bleeding","Colon polyp",

    # Hematology & Oncology (18)
    "Iron-deficiency anemia","Vitamin B12 deficiency","Sickle cell disease","Thalassemia",
    "Hemophilia","Leukemia","Lymphoma","Hodgkin lymphoma","Non-Hodgkin lymphoma",
    "Multiple myeloma","Breast cancer","Lung cancer","Colorectal cancer","Prostate cancer",
    "Cervical cancer","Ovarian cancer","Melanoma","Chemotherapy",

    # Neurology & Psychiatry (16)
    "Epilepsy","Migraine","Parkinson's disease","Alzheimer's disease","Multiple sclerosis",
    "Guillain–Barré syndrome","Myasthenia gravis","Peripheral neuropathy",
    "Depression (mood)","Anxiety disorder","Bipolar disorder","Schizophrenia",
    "Delirium","Ischemic stroke","Subarachnoid hemorrhage","Intracerebral hemorrhage",

    # Rheumatology & Immunology (12)
    "Rheumatoid arthritis","Osteoarthritis","Gout","Systemic lupus erythematosus",
    "Ankylosing spondylitis","Psoriatic arthritis","Vasculitis","Giant-cell arteritis",
    "Polymyalgia rheumatica","Sjogren's syndrome","Scleroderma","Sarcoidosis",

    # Nephrology & Urology (10)
    "Chronic kidney disease","Acute kidney injury","Glomerulonephritis",
    "Nephrotic syndrome","Kidney stone","Benign prostatic hyperplasia",
    "Prostatitis","Hematuria","Urinary incontinence","Hemodialysis",

    # Obstetrics & Gynecology (8)
    "Preeclampsia","Gestational diabetes","Preterm birth","Endometriosis",
    "Uterine fibroid","Ectopic pregnancy","Postpartum hemorrhage","Cervical cancer screening",

    # Dermatology (6)
    "Psoriasis","Atopic dermatitis","Acne vulgaris","Cellulitis","Melanoma","Urticaria",

    # Drugs & Therapeutics (18)
    "Aspirin","Clopidogrel","Warfarin","Direct oral anticoagulant","Heparin","Labetalol",
    "Nifedipine","Amoxicillin","Ciprofloxacin","Doxycycline","Ceftriaxone","Vancomycin",
    "Oseltamivir","Remdesivir","Naloxone","Morphine","Proton pump inhibitor","Inhaled corticosteroid",

    # Diagnostics & Procedures (16)
    "Electrocardiography","Echocardiography","Spirometry","Colonoscopy","Endoscopy",
    "Ultrasound","Computed tomography","Magnetic resonance imaging","Positron emission tomography",
    "Chest X-ray","Lumbar puncture","Biopsy","Arterial blood gas","Blood culture",
    "Complete blood count","Basic life support"
]
# assert len(TITLES) == 150, f"Expected 150 titles, found {len(TITLES)}"

# --------------------------------
# Wikipedia client and helpers
# --------------------------------
wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="healthcare-rag-bot/0.1 (shangkaul@yourdomain.com)"
)

STOP_SECTIONS = [
    "See also","References","External links","Further reading","Bibliography","Notes","Citations"
]

def strip_tail_sections(text: str) -> str:
    """
    Remove trailing non-content sections like 'References' or 'See also'.
    Looks for '== Section ==' style headings and truncates at first stop section.
    """
    # Find all top-level section headings
    pattern = re.compile(r"\n==\s*(.+?)\s*==\n")
    positions = [(m.start(), m.group(1).strip()) for m in pattern.finditer(text)]
    if not positions:
        return text

    cut_pos = None
    for pos, name in positions:
        if any(name.lower().startswith(s.lower()) for s in STOP_SECTIONS):
            cut_pos = pos
            break
    return text if cut_pos is None else text[:cut_pos].rstrip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Sentence-aware chunking with token-approx by words (fast, robust)."""
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks, current, length = [], [], 0
    for sent in sentences:
        words = sent.split()
        if length + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            # keep overlap
            current = current[-overlap:]
            length  = len(current)
        current.extend(words)
        length += len(words)
    if current:
        chunks.append(" ".join(current))
    # filter tiny leftovers
    return [c for c in chunks if len(c.split()) > 50]

def fetch_page_text(title: str, retries: int = 3, delay: float = 0.8) -> str:
    """Fetch page text with simple retry and rate limiting."""
    for attempt in range(1, retries + 1):
        page = wiki.page(title)
        if page.exists():
            txt = page.text or ""
            if txt.strip():
                return txt
        time.sleep(delay * attempt)  # backoff
    return ""

# --------------------------------
# Main collection & writing
# --------------------------------
out_path = "data/wiki_corpus.jsonl"
seen_ids = set()
total_chunks = 0
missing = []

with open(out_path, "w", encoding="utf-8") as f:
    for title in tqdm(TITLES, desc="Collecting Wikipedia medical corpus"):
        raw = fetch_page_text(title)
        if not raw:
            missing.append(title)
            continue

        cleaned = strip_tail_sections(raw)
        chunks = chunk_text(cleaned, chunk_size=800, overlap=150)

        base_id = title.replace(" ", "_")
        for i, chunk in enumerate(chunks):
            doc_id = f"{base_id}_{i}"
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            record = {
                "id": doc_id,
                "title": title,
                "source": "wikipedia",
                "chunk_index": i,
                "text": chunk
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chunks += 1

print(f"Saved {total_chunks} chunks from {len(TITLES) - len(missing)} / {len(TITLES)} articles to {out_path}")
if missing:
    print("Missing or empty pages (check spelling/disambiguation):")
    for t in missing:
        print(" -", t)
