import streamlit as st
import os, io, re, time, datetime, random, sqlite3
from typing import List, Dict, Tuple
import pdfplumber
import docx2txt
try:
    from rapidfuzz import fuzz
except Exception:
    from difflib import SequenceMatcher
    def _ratio(a,b): return int(SequenceMatcher(None, a, b).ratio()*100)
    class fuzz:
        @staticmethod
        def token_set_ratio(a,b):
            return _ratio(a,b)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTE_AVAILABLE = True
except Exception:
    SENTE_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64

SAMPLE_JD_1 = "/mnt/data/sample_jd_1.pdf"
SAMPLE_JD_2 = "/mnt/data/sample_jd_2.pdf"
SAMPLE_RESUME = "/mnt/data/resume - 1.pdf"

def extract_text_from_pdf(path: str) -> str:
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception:
        return ""

def extract_text_from_docx(path: str) -> str:
    try:
        return docx2txt.process(path) or ""
    except Exception:
        return ""

def extract_text(path_or_buf) -> str:
    if not path_or_buf:
        return ""
    if hasattr(path_or_buf, "read"):
        data = path_or_buf.read()
        name = getattr(path_or_buf, "name", "")
        tmp = f"./tmp_{int(time.time()*1000)}_{random.randint(0,999)}"
        ext = os.path.splitext(name)[1].lower()
        tmp_path = tmp + ext if ext else tmp + ".pdf"
        with open(tmp_path, "wb") as f:
            f.write(data)
        txt = extract_text_from_pdf(tmp_path) if tmp_path.endswith(".pdf") else extract_text_from_docx(tmp_path)
        try:
            os.remove(tmp_path)
        except: pass
        return txt
    else:
        path = str(path_or_buf)
        if path.lower().endswith(".pdf"):
            return extract_text_from_pdf(path)
        elif path.lower().endswith(".docx"):
            return extract_text_from_docx(path)
        elif path.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except:
                return ""
        else:
            return ""

def fuzzy_match(skill: str, text: str, thresh:int=70) -> Tuple[bool,int]:
    if not skill or not text:
        return False, 0
    score = fuzz.token_set_ratio(skill.lower(), text.lower())
    return (score >= thresh), int(score)

def tfidf_similarity(a: str, b: str) -> float:
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0:1], X[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0

SENTE_MODEL = None
if SENTE_AVAILABLE:
    try:
        SENTE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        SENTE_MODEL = None

def sente_similarity(a: str, b: str) -> float:
    if not SENTE_MODEL:
        return 0.0
    a_emb = SENTE_MODEL.encode(a)
    b_emb = SENTE_MODEL.encode(b)
    denom = (np.linalg.norm(a_emb)*np.linalg.norm(b_emb))
    if denom == 0:
        return 0.0
    return float((a_emb @ b_emb) / denom)

def evaluate(resume_text: str, jd_text: str, must_have: List[str], good_have: List[str],
             hard_w: float, soft_w: float, fuzzy_thresh: int, embed_method: str) -> Dict:
    resume_text = resume_text or ""
    jd_text = jd_text or ""
    matched_must = []
    missing_must = []
    for s in must_have:
        ok, sc = fuzzy_match(s, resume_text, thresh=fuzzy_thresh)
        if ok: matched_must.append((s, sc))
        else: missing_must.append((s, sc))
    matched_good = []
    missing_good = []
    for s in good_have:
        ok, sc = fuzzy_match(s, resume_text, thresh=fuzzy_thresh)
        if ok: matched_good.append((s, sc))
        else: missing_good.append((s, sc))
    total_must = max(1, len(must_have))
    total_good = max(1, len(good_have))
    must_ratio = len(matched_must)/total_must
    good_ratio = len(matched_good)/total_good if good_have else 0.0
    hard_score = (0.75 * must_ratio + 0.25 * good_ratio) * 100
    soft_sim = 0.0
    method_used = "tfidf"
    try:
        if embed_method == "sente" and SENTE_MODEL:
            import numpy as np
            a = SENTE_MODEL.encode(jd_text)
            b = SENTE_MODEL.encode(resume_text)
            denom = (np.linalg.norm(a)*np.linalg.norm(b))
            soft_sim = float((a @ b) / denom) if denom!=0 else 0.0
            method_used = "sentence-transformer"
        else:
            soft_sim = tfidf_similarity(jd_text, resume_text)
            method_used = "tfidf"
    except Exception:
        soft_sim = tfidf_similarity(jd_text, resume_text)
        method_used = "tfidf"
    soft_score = float(soft_sim * 100)
    final_score = hard_w*hard_score + soft_w*soft_score
    final_score = max(0.0, min(100.0, final_score))
    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"
    suggestions = []
    for s, _ in missing_must + missing_good:
        suggestions.append(f"Learn or demonstrate '{s}' (courses, projects, or certifications).")
    return {
        "hard_score": round(hard_score,2),
        "soft_score": round(soft_score,2),
        "final_score": round(final_score,2),
        "verdict": verdict,
        "matched_must": matched_must,
        "missing_must": missing_must,
        "matched_good": matched_good,
        "missing_good": missing_good,
        "method_used": method_used,
        "suggestions": suggestions
    }

def extract_skills_from_jd(jd_text: str, max_skills:int=12) -> Tuple[List[str], List[str]]:
    jd = (jd_text or "").lower()
    lines = jd.splitlines()
    must = []
    good = []
    for ln in lines:
        if any(k in ln for k in ["must have","must-have","required","requirements","skills required","must have:","must:"]):
            parts = re.split(r"[:,;-]| and |,|;|/|\(|\)", ln)
            for p in parts:
                p = p.strip()
                if 3 <= len(p) <= 40:
                    must.append(p)
        if any(k in ln for k in ["good to have","nice to have","preferred","advantageous","desirable"]):
            parts = re.split(r"[:,;-]| and |,|;|/|\(|\)", ln)
            for p in parts:
                p = p.strip()
                if 3 <= len(p) <= 40:
                    good.append(p)
    if not must:
        tokens = re.findall(r"[A-Za-z+#]+(?:\s[A-Za-z+#]+)?", jd_text)
        stop = set(["experience","years","required","qualification","role","responsibility","work","with","and"])
        filtered = [t.strip() for t in tokens if t.strip().lower() not in stop and len(t.strip())>2]
        freq = {}
        for t in filtered:
            freq[t] = freq.get(t,0) + 1
        sorted_tokens = sorted(freq.items(), key=lambda x: -x[1])
        must = [t[0] for t in sorted_tokens[:max_skills]]
    def clean_list(lst):
        out=[]
        for x in lst:
            x = x.strip().strip(':').strip()
            if x and x not in out:
                out.append(x)
        return out
    return clean_list(must), clean_list(good)

DB_PATH = "results.db"
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_file TEXT,
            candidate_name TEXT,
            candidate_email TEXT,
            job_title TEXT,
            hard_score REAL,
            soft_score REAL,
            final_score REAL,
            verdict TEXT,
            missing_items TEXT,
            suggestions TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    return conn, cur

def insert_eval(cur, conn, rec: Dict):
    cur.execute("""
        INSERT INTO evaluations (resume_file, candidate_name, candidate_email, job_title,
                                 hard_score, soft_score, final_score, verdict, missing_items, suggestions, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        rec.get("resume_file"), rec.get("candidate_name"), rec.get("candidate_email"), rec.get("job_title"),
        rec.get("hard_score"), rec.get("soft_score"), rec.get("final_score"), rec.get("verdict"),
        rec.get("missing_items"), rec.get("suggestions"), rec.get("timestamp")
    ))
    conn.commit()

st.set_page_config(page_title="Automated Resume Relevance Check", layout="wide")
st.title("Theme 2 — Automated Resume Relevance Check System")

with st.sidebar:
    st.header("Configuration")
    embed_choice = st.selectbox("Embedding method (if sentence-transformers not installed TF-IDF is used)", options=("auto","sente","tfidf"))
    if embed_choice == "sente" and not SENTE_MODEL:
        st.warning("sentence-transformers not available; TF-IDF will be used.")
    hard_weight = st.slider("Hard match weight", 0.0, 1.0, 0.6, step=0.05)
    soft_weight = st.slider("Soft match weight", 0.0, 1.0, 0.4, step=0.05)
    if hard_weight + soft_weight != 1.0:
        st.caption("Weights will be normalized.")
    fuzzy_thresh = st.slider("Fuzzy threshold (0-100)", 40, 95, 70)
    st.markdown("---")
    st.write("Sample files uploaded by you (auto-detected):")
    for p in (SAMPLE_JD_1, SAMPLE_JD_2, SAMPLE_RESUME):
        st.write(f"- {p}")

col1, col2 = st.columns([2,3])
with col1:
    st.subheader("1) Job Description (JD)")
    jd_mode = st.radio("JD source", ("Paste JD text", "Upload JD file", "Use sample JD (1)", "Use sample JD (2)"), index=2)
    jd_text = ""
    if jd_mode == "Paste JD text":
        jd_text = st.text_area("Paste the Job Description here", height=220)
    elif jd_mode == "Upload JD file":
        uploaded_jd = st.file_uploader("Upload JD (pdf/docx/txt)", type=["pdf","docx","txt"], key="jd_upload")
        if uploaded_jd:
            jd_text = extract_text(uploaded_jd)
            st.success("JD loaded.")
    elif jd_mode == "Use sample JD (1)":
        jd_text = extract_text(SAMPLE_JD_1)
        st.write("Loaded sample_jd_1.pdf")
    else:
        jd_text = extract_text(SAMPLE_JD_2)
        st.write("Loaded sample_jd_2.pdf")
    if jd_text:
        st.markdown("### Parsed JD (preview)")
        st.write(jd_text[:2000] + ("..." if len(jd_text)>2000 else ""))
        must_list, good_list = extract_skills_from_jd(jd_text)
        st.markdown("**Auto-extracted Must-have (editable):**")
        must_input = st.text_area("Must-have skills (comma-separated)", value=", ".join(must_list), height=80)
        st.markdown("**Auto-extracted Good-to-have (editable):**")
        good_input = st.text_area("Good-to-have skills (comma-separated)", value=", ".join(good_list), height=80)
    else:
        st.info("No JD loaded yet.")
        must_input = ""
        good_input = ""

with col2:
    st.subheader("2) Resumes")
    resume_mode = st.radio("Resumes source", ("Upload resumes", "Use sample resume"), index=1)
    uploaded_files = []
    if resume_mode == "Upload resumes":
        uploaded_files = st.file_uploader("Upload one or more resumes (pdf/docx/txt)", accept_multiple_files=True)
    else:
        st.write(f"Using sample resume: {SAMPLE_RESUME}")
        uploaded_files = [SAMPLE_RESUME]
    st.markdown("You can set weights and threshold from the left panel, then click Run.")

run_button = st.button("Run Relevance Check")
conn, cur = init_db()

if run_button:
    if not jd_text:
        st.error("Please provide a Job Description (JD) first.")
    elif not uploaded_files:
        st.error("Please upload or choose at least one resume.")
    else:
        total_w = hard_weight + soft_weight
        if total_w == 0:
            hard_w, soft_w = 0.6, 0.4
        else:
            hard_w, soft_w = hard_weight/total_w, soft_weight/total_w
        embed_method = "tfidf"
        if embed_choice == "auto":
            embed_method = "sente" if SENTE_MODEL else "tfidf"
        elif embed_choice == "sente":
            embed_method = "sente" if SENTE_MODEL else "tfidf"
        else:
            embed_method = "tfidf"
        must_list = [s.strip() for s in (must_input or "").split(",") if s.strip()]
        good_list = [s.strip() for s in (good_input or "").split(",") if s.strip()]
        progress = st.progress(0)
        results = []
        N = len(uploaded_files)
        for i, up in enumerate(uploaded_files):
            if hasattr(up, "name"):
                fname = up.name
            else:
                fname = os.path.basename(str(up))
            txt = extract_text(up)
            email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", txt)
            email = email_match.group(0) if email_match else ""
            name = ""
            first_lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            if first_lines:
                name = first_lines[0][:120]
            eval_res = evaluate(resume_text=txt, jd_text=jd_text, must_have=must_list, good_have=good_list,
                                 hard_w=hard_w, soft_w=soft_w, fuzzy_thresh=fuzzy_thresh, embed_method=embed_method)
            missing_items = [m[0] for m in eval_res["missing_must"]] + [g[0] for g in eval_res["missing_good"]]
            suggestions = eval_res["suggestions"]
            rec = {
                "resume_file": fname,
                "candidate_name": name,
                "candidate_email": email,
                "job_title": (jd_text.splitlines()[0][:120] if jd_text else "Unknown"),
                "hard_score": eval_res["hard_score"],
                "soft_score": eval_res["soft_score"],
                "final_score": eval_res["final_score"],
                "verdict": eval_res["verdict"],
                "missing_items": "; ".join(missing_items),
                "suggestions": " ".join(suggestions),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            insert_eval(cur, conn, rec)
            results.append(rec)
            progress.progress(int((i+1)/N * 100))
        st.success("Evaluation completed — results saved to local DB.")
        df = pd.DataFrame(results).sort_values("final_score", ascending=False)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="relevance_results.csv">Download Results CSV</a>', unsafe_allow_html=True)

st.markdown("---")
st.subheader("Dashboard / Stored Evaluations")
try:
    df_db = pd.read_sql_query("SELECT * FROM evaluations ORDER BY final_score DESC", conn)
    st.write(f"Total evaluations stored: {len(df_db)}")
    if not df_db.empty:
        c1, c2 = st.columns([2,1])
        with c1:
            job_filter = st.selectbox("Filter by job title (first line)", options=["All"] + sorted(df_db.job_title.unique().tolist()))
        with c2:
            score_range = st.slider("Score range", 0.0, 100.0, (0.0, 100.0))
        out = df_db.copy()
        if job_filter != "All":
            out = out[out.job_title == job_filter]
        out = out[(out.final_score >= score_range[0]) & (out.final_score <= score_range[1])]
        st.dataframe(out)
        st.markdown("**Verdict counts**")
        st.bar_chart(df_db.verdict.value_counts())
        st.markdown("**Average score**")
        st.metric("Average final score", f"{df_db.final_score.mean():.2f}")
        csv2 = out.to_csv(index=False)
        b64 = base64.b64encode(csv2.encode()).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="dashboard_filtered.csv">Download Filtered CSV</a>', unsafe_allow_html=True)
    else:
        st.info("No evaluations stored yet — run one.")
except Exception as e:
    st.error(f"An error occurred while accessing the database: {e}")