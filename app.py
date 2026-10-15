"""Khmer OCR Demo — Streamlit interface.

This file is presentation only. Recognition is delegated to the existing
Recognizer in inference_three.py with its preprocessing and CTC decoding
untouched, so predictions here are the same ones the project produces
elsewhere. Nothing is cached, stubbed or hardcoded.
"""
from __future__ import annotations

import io
import json
import time
import unicodedata
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

from inference_three import MultiRecognizer

REPO_ROOT = Path(__file__).resolve().parent
DEMO_ZIP = REPO_ROOT / "demo_dataset.zip"
DEMO_DIR = REPO_ROOT / ".demo_cache"
HF_MODELS_REPO = "Seoul12/khmer-ocr-three-architectures"
HF_DEMO_REPO = "Seoul12/khmer-ocr-selected-demo"
HF_DEMO_FILE = "Expanded_Khmer_OCR_Demo.zip"
# Built locally by screen_handwriting.py. DEMO_GUIDE.md points at
# Seoul12/khmer-ocr-handwriting-demo instead, but that repo holds only a
# .gitattributes -- the archive it describes was never uploaded.
HAND_DIR = Path(__file__).resolve().parent / "handwriting_demo"

# The three saved architectures. transformer.pt is the same step-87,000
# checkpoint published as Seoul12/khmer-ocr-architecture3-best.
COMPARE_ALL = "Compare all"

MODELS = {
    "Transformer": {"key": "transformer", "file": "transformer.pt",
                    "label": "Transformer encoder + CTC", "step": "87,000"},
    "BiLSTM":      {"key": "bilstm", "file": "bilstm.pt",
                    "label": "BiLSTM + CTC", "step": "58,000"},
    "BiGRU":       {"key": "bigru", "file": "bigru.pt",
                    "label": "BiGRU + CTC", "step": "8,500"},
}
GITHUB_URL = "https://github.com/SeoulDoriot/khmerOCR"

st.set_page_config(page_title="Khmer OCR Demo", page_icon="◦",
                   layout="centered", initial_sidebar_state="collapsed")

st.html(r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Khmer:wght@400;500&display=swap">
<style>
:root{
  --bg:#FFFFFF; --soft:#F7F7F5; --soft-2:#F5F5F7; --raise:#FCFCFB;
  --ink:#1D1D1F; --ink-2:#6E6E73; --ink-3:#95959B; --line:#E7E7E5; --line-2:#DBDBD7;
  --ok:#1A7F4B; --ok-bg:#EDF7F1; --warn:#9A5B12; --warn-bg:#FBF3E8;
  /* one accent, used only for the primary action, the active tab and focus */
  --accent:#2C5FE8; --accent-ink:#1E47B5; --accent-soft:#EEF2FE;
  --ease:cubic-bezier(.22,.61,.36,1);
  --shadow-1:0 1px 2px rgba(20,20,22,.04), 0 1px 1px rgba(20,20,22,.03);
  --shadow-2:0 4px 14px rgba(20,20,22,.07), 0 1px 3px rgba(20,20,22,.04);
  --shadow-3:0 14px 40px rgba(20,20,22,.09), 0 2px 8px rgba(20,20,22,.04);
}
html, body, [class*="css"]{
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink); -webkit-font-smoothing:antialiased;
}
.stApp{background:var(--bg)}
div#MainMenu, footer, header[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"], [data-testid="stDecoration"]{display:none!important}
/* ONE content width for every block, so nothing sits off-centre against
   anything else. Streamlit's column rows cannot be capped individually
   without breaking alignment with the blocks above and below them. */
.block-container{max-width:1060px;padding:0 clamp(18px,4vw,40px) 120px;margin:0 auto}

@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes popIn{0%{opacity:0;transform:translateY(18px) scale(.985)}
  60%{opacity:1}100%{opacity:1;transform:none}}
@keyframes sweep{0%{background-position:-220% 0}100%{background-position:220% 0}}
@keyframes orbit{to{transform:rotate(360deg)}}
@keyframes breathe{0%,100%{opacity:.35}50%{opacity:.9}}

/* ---------- header ---------- */
.topbar{display:flex;align-items:center;justify-content:space-between;height:74px;
  border-bottom:1px solid var(--line);margin-bottom:56px;
  position:sticky;top:0;z-index:50;background:rgba(255,255,255,.86);
  backdrop-filter:saturate(180%) blur(14px);
  animation:fadeIn .5s var(--ease) both}
.brand{font-size:19px;font-weight:600;letter-spacing:-.02em;display:flex;align-items:center;gap:10px}
.brand .dot{width:9px;height:9px;border-radius:50%;background:var(--accent);
  animation:breathe 3.4s ease-in-out infinite}
.ghlink{font-size:15px;color:var(--ink-2);text-decoration:none;
  transition:color .2s var(--ease)}
.ghlink:hover{color:var(--ink)}

/* ---------- hero ---------- */
/* Left-aligned and two lines deep. Centred text with four stacked grey
   paragraphs gave every line the same weight, so nothing led. */
.hero{text-align:left;margin-bottom:40px;max-width:680px}
.hero h1{font-size:clamp(34px,4.6vw,46px);line-height:1.06;font-weight:600;
  letter-spacing:-.032em;margin:0 0 14px;animation:fadeUp .62s var(--ease) both}
.hero .sub{font-size:17px;color:var(--ink-2);margin:0;line-height:1.6;
  font-weight:400;animation:fadeUp .62s .07s var(--ease) both}
.hero .supp,.hero .helper{display:none}
.hero .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px;
  animation:fadeUp .62s .13s var(--ease) both}
.hero .chips span{font-size:12.5px;font-weight:500;color:var(--ink-2);
  background:var(--soft);border:1px solid var(--line);border-radius:999px;
  padding:6px 12px;letter-spacing:.005em}

/* ---------- section headings ---------- */
.sec{font-size:18px;font-weight:600;letter-spacing:-.018em;color:var(--ink);margin:0 0 10px}
.sec-sub{font-size:14.5px;color:var(--ink-2);margin:0 0 22px;line-height:1.6}
.st-key-more,.st-key-less{display:flex!important;justify-content:center!important;
  width:100%!important;flex:1 1 100%!important}
/* when both buttons show they sit in half-width columns; pull them together
   around the centre line instead of floating in each half */
[data-testid="stColumn"] .st-key-more,
[data-testid="stColumn"] .st-key-more .stButton{justify-content:flex-end!important}
[data-testid="stColumn"] .st-key-less,
[data-testid="stColumn"] .st-key-less .stButton{justify-content:flex-start!important}
[data-testid="stColumn"] .st-key-more{padding-right:7px}
[data-testid="stColumn"] .st-key-less{padding-left:7px}
.st-key-more .stButton,.st-key-less .stButton{display:flex!important;
  justify-content:center!important;width:100%!important}
.st-key-more .stButton>button,.st-key-less .stButton>button{
  width:auto!important;min-width:200px;max-width:280px;flex:0 0 auto!important}
.stElementContainer:has(> .stButton > button):not([class*="st-key-ex"]) .stButton{
  display:flex;justify-content:center}
.galcount{text-align:center;font-size:12.5px;color:var(--ink-3);margin:4px 0 12px;
  font-variant-numeric:tabular-nums}
.pickerlabel{font-size:13px;font-weight:500;color:var(--ink-2);margin:0 0 8px;
  letter-spacing:.005em}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]{
  border-radius:10px!important;font-size:14px!important;font-weight:500!important;
  transition:background .18s var(--ease),color .18s var(--ease)!important}
[data-testid="stButtonGroup"] button[data-selected="true"]{
  background:var(--accent-soft)!important;color:var(--accent-ink)!important}
.droptitle{display:none}

/* ---------- uploader ---------- */
[data-testid="stFileUploader"]{background:var(--soft);
  border:1.5px dashed var(--line-2);
  border-radius:18px;padding:22px 24px;
  transition:border-color .25s var(--ease),background .25s var(--ease),
    box-shadow .25s var(--ease),transform .25s var(--ease);
  animation:fadeUp .62s .24s var(--ease) both}
[data-testid="stFileUploader"]:hover{border-color:var(--accent);background:var(--accent-soft);
  box-shadow:var(--shadow-2);transform:translateY(-1px)}
[data-testid="stFileUploader"] label{display:none}
[data-testid="stFileUploader"] section{background:transparent!important;border:none!important;
  padding:0!important;display:flex;flex-direction:column;align-items:center;gap:0}
[data-testid="stFileUploaderDropzoneInstructions"]{display:none!important}
[data-testid="stFileUploader"] section::before{content:"Drop a cropped Khmer text line here";
  display:block;text-align:center;font-size:15px;font-weight:500;color:var(--ink);
  margin:0 0 4px}
[data-testid="stFileUploader"] section::after{content:"PNG or JPG · one line reads best";
  display:block;text-align:center;font-size:12.5px;color:var(--ink-3);
  margin-top:14px;letter-spacing:.01em}
[data-testid="stFileUploader"] section > div{margin-top:12px}
[data-testid="stFileUploader"] section > div{width:100%;display:flex;justify-content:center}
[data-testid="stFileUploader"] section button{margin:0 auto!important;border-radius:11px!important;
  background:var(--bg)!important;color:var(--ink)!important;border:1px solid var(--line-2)!important;
  font-weight:500!important;font-size:15px!important;padding:11px 24px!important;
  min-height:46px!important;box-shadow:var(--shadow-1)!important;
  transition:transform .18s var(--ease),box-shadow .18s var(--ease),border-color .18s var(--ease)!important}
[data-testid="stFileUploader"] section button:hover{border-color:var(--line-2)!important;
  box-shadow:var(--shadow-2)!important;transform:translateY(-1px)}
[data-testid="stFileUploader"] section button:active{transform:translateY(0) scale(.985)}
[data-testid="stFileUploader"] section button > *{display:none!important}
[data-testid="stFileUploader"] section button::after{content:"Browse files";
  display:inline-block;font-size:15px;font-weight:500;color:var(--ink)}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"]{display:none}

/* ---------- buttons ---------- */
.stButton > button{border-radius:12px;font-weight:500;font-size:15px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:11px 20px;
  box-shadow:var(--shadow-1);
  transition:transform .18s var(--ease),box-shadow .18s var(--ease),
    border-color .18s var(--ease),background .18s var(--ease)}
.stButton > button:hover{background:var(--raise);border-color:var(--line-2);
  color:var(--ink);box-shadow:var(--shadow-2);transform:translateY(-1px)}
.stButton > button:active{transform:translateY(0) scale(.985);box-shadow:var(--shadow-1)}
.stButton > button:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
.stButton > button[kind="primary"]{background:var(--ink);color:#fff;border-color:var(--ink);
  height:52px;font-size:16px;font-weight:500;letter-spacing:-.01em;box-shadow:var(--shadow-2);
  background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important}
.stButton > button[kind="primary"]:hover{background:var(--accent-ink)!important;
  border-color:var(--accent-ink)!important}
.stButton > button[kind="primary"]:hover{background:#000;border-color:#000;color:#fff;
  box-shadow:var(--shadow-3);transform:translateY(-1.5px)}
.stButton > button[kind="primary"]:active{transform:translateY(0) scale(.99)}
.stDownloadButton > button{border-radius:12px;font-size:14px;font-weight:500;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);padding:9px 18px;
  box-shadow:var(--shadow-1);
  transition:transform .18s var(--ease),box-shadow .18s var(--ease),border-color .18s var(--ease)}
.stDownloadButton > button:hover{background:var(--raise);border-color:var(--line-2);
  color:var(--ink);box-shadow:var(--shadow-2);transform:translateY(-1px)}
.stDownloadButton > button:active{transform:translateY(0) scale(.985)}

/* ---------- preview ---------- */
[data-testid="stImage"]{background:var(--soft);border:1px solid var(--line);
  border-radius:20px;padding:24px;animation:popIn .5s var(--ease) both;
  transition:box-shadow .25s var(--ease)}
[data-testid="stImage"]:hover{box-shadow:var(--shadow-2)}
/* gallery tile: image box of a fixed height, caption button at the bottom, so
   every card in a row is the same height whatever its image's aspect ratio */
[class*="st-key-tile"]{display:flex;flex-direction:column;height:100%}
[class*="st-key-tile"] [data-testid="stImage"]{padding:14px;border-radius:16px;
  height:108px;width:100%;display:flex;align-items:center;justify-content:center;
  overflow:hidden;
  transition:transform .2s var(--ease),box-shadow .2s var(--ease),border-color .2s var(--ease)}
[class*="st-key-tile"] [data-testid="stImage"]:hover{transform:translateY(-2px);
  box-shadow:var(--shadow-2);border-color:var(--line-2)}
[class*="st-key-tile"] [data-testid="stImage"] > div,
[class*="st-key-tile"] [data-testid="stImage"] figure{width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;margin:0}
[class*="st-key-tile"] [data-testid="stImage"] img{max-height:80px;max-width:100%;
  width:auto;height:auto;object-fit:contain}
[class*="st-key-tile"] .stButton{margin-top:auto;padding-top:10px}
[class*="st-key-tile"] .stButton > button{width:100%}
[data-testid="stImage"] img{border-radius:12px;background:#fff;box-shadow:var(--shadow-1)}
.filemeta{font-size:13px;color:var(--ink-3);margin-top:12px;text-align:center;
  overflow-wrap:anywhere;animation:fadeIn .4s .1s var(--ease) both}

/* ---------- result ---------- */
.resultbox{background:var(--bg);border:1px solid var(--line);border-radius:20px;
  padding:38px 34px 26px;box-shadow:var(--shadow-2);
  animation:popIn .55s var(--ease) both;position:relative;overflow:hidden}
.resultbox::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,transparent,var(--line-2),transparent)}
.resultbox .khmer, p.khmer{font-family:"Noto Sans Khmer",Inter,sans-serif!important;
  font-size:clamp(23px,3.1vw,32px)!important;line-height:2.0!important;
  color:var(--ink)!important;overflow-wrap:anywhere;margin:0!important;
  font-weight:400!important;letter-spacing:-.005em;
  animation:fadeUp .5s .12s var(--ease) both}
.resultfoot{display:flex;align-items:center;justify-content:space-between;gap:12px;
  margin-top:26px;padding-top:18px;border-top:1px solid var(--line);flex-wrap:wrap;
  animation:fadeIn .5s .26s var(--ease) both}
.resultfoot .tag{font-size:12px;color:var(--ink-3);letter-spacing:.02em}
.resultfoot .tag.mono{font-variant-numeric:tabular-nums}
.resultbox.compact{padding:24px 26px 18px;margin-bottom:14px}
.resultbox.compact .khmer, .resultbox.compact p.khmer{
  font-size:clamp(19px,2.3vw,25px)!important;line-height:1.95!important}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:14px;margin-top:6px}
table.cmp{border-collapse:collapse;width:100%;font-size:13.5px;min-width:620px}
table.cmp th,table.cmp td{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line)}
table.cmp th{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);font-weight:500;background:var(--soft)}
table.cmp tr:last-child td{border-bottom:0}
table.cmp td.m{font-weight:500;white-space:nowrap}
table.cmp td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
table.cmp td.km-cell{font-family:"Noto Sans Khmer",Inter,sans-serif;font-size:15px;
  line-height:1.9;min-width:220px}
.tablenote{font-size:12.5px;color:var(--ink-3);margin:12px 0 0;line-height:1.6}
table.cmp td.num.ok{color:var(--ok);font-weight:500}
table.cmp td.num.no{color:var(--bad)}
.reflabel{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 7px}
.refbox{background:var(--soft);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;margin-bottom:16px}
.khmer-inline{font-family:"Noto Sans Khmer",Inter,sans-serif;font-size:17px;
  line-height:1.95;color:var(--ink)}

/* ---------- loading ---------- */
.loadwrap{border:1px solid var(--line);border-radius:20px;background:var(--bg);
  padding:40px 34px;display:flex;align-items:center;gap:18px;
  box-shadow:var(--shadow-1);animation:fadeIn .3s var(--ease) both}
.spinner{width:22px;height:22px;flex:0 0 22px;border-radius:50%;
  border:2px solid var(--line);border-top-color:var(--ink);
  animation:orbit .75s linear infinite}
.loadtxt .t{font-size:15.5px;font-weight:500;color:var(--ink);margin:0 0 4px;letter-spacing:-.01em}
.loadtxt .d{font-size:13.5px;color:var(--ink-2);margin:0}
.skeleton{height:14px;border-radius:7px;margin-top:22px;
  background:linear-gradient(90deg,var(--soft) 25%,#EDEDEA 50%,var(--soft) 75%);
  background-size:220% 100%;animation:sweep 1.4s linear infinite}

/* ---------- badges ---------- */
.badge{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:500;
  padding:6px 13px;border-radius:999px;background:var(--soft-2);color:var(--ink-2);
  animation:fadeIn .4s var(--ease) both}
.badge.sel{background:var(--warn-bg);color:var(--warn)}
.badge.sel::before{content:"";width:5px;height:5px;border-radius:50%;background:currentColor}

/* ---------- metrics ---------- */
.metricrow{display:flex;align-items:baseline;justify-content:space-between;
  padding:14px 0;border-bottom:1px solid var(--line);animation:fadeIn .4s var(--ease) both}
.metricrow:last-child{border-bottom:0}
.metricrow .k{font-size:14px;color:var(--ink-2)}
.metricrow .v{font-size:15px;font-weight:500;font-variant-numeric:tabular-nums}
.metricrow .v.good{color:var(--ok)}

/* ---------- notes ---------- */
.note{background:var(--soft);border-radius:20px;padding:28px 30px;
  transition:background .25s var(--ease)}
.note:hover{background:var(--soft-2)}
.note h4{font-size:15.5px;font-weight:600;margin:0 0 9px;letter-spacing:-.012em}
.note p{font-size:14px;color:var(--ink-2);margin:0;line-height:1.7;max-width:62ch}
.about{font-size:13.5px;color:var(--ink-2);line-height:2.05}
.about b{color:var(--ink);font-weight:500}
.foot{text-align:center;font-size:13px;color:var(--ink-3);padding-top:32px;
  border-top:1px solid var(--line)}

/* ---------- streamlit odds and ends ---------- */
.stTextArea textarea{border-radius:14px!important;border-color:var(--line)!important;
  font-family:"Noto Sans Khmer",Inter,sans-serif!important;font-size:16px!important;
  transition:border-color .2s var(--ease),box-shadow .2s var(--ease)!important}
.stTextArea textarea:focus{border-color:var(--ink)!important;
  box-shadow:0 0 0 3px rgba(29,29,31,.07)!important}
[data-testid="stExpander"]{border:none!important;background:transparent!important}
[data-testid="stExpander"] details{border:1px solid var(--line)!important;border-radius:16px!important;
  background:var(--bg)!important;transition:border-color .2s var(--ease),box-shadow .2s var(--ease)!important}
[data-testid="stExpander"] details:hover{border-color:var(--line-2)!important;
  box-shadow:var(--shadow-1)!important}
[data-testid="stExpander"] summary{font-size:14px!important;font-weight:500!important;
  padding:16px 20px!important}
hr{border:none;border-top:1px solid var(--line);margin:48px 0}
.stAlert{border-radius:16px;animation:popIn .4s var(--ease) both}
[data-testid="stAlertContainer"]{border-radius:16px}

/* Streamlit lays columns out as a non-wrapping flex row; let them wrap */
[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:14px!important;
  align-items:stretch!important}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:0}
@media (max-width:880px){
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    flex:1 1 calc(50% - 14px)!important;min-width:calc(50% - 14px)!important}
}
@media (max-width:560px){
  /* One column, gallery included. A cropped text line is 512px wide: at 2-up on
     a phone each card is ~160px and the Khmer is too small to read, so the
     gallery goes full width here and the image box grows with it. */
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{
    flex:1 1 100%!important;min-width:100%!important}
  [class*="st-key-tile"] [data-testid="stImage"]{height:92px;padding:12px}
  [class*="st-key-tile"] [data-testid="stImage"] img{max-height:68px}
}
@media (max-width:640px){
  .block-container{padding:0 18px 88px}
  .topbar{height:64px;margin-bottom:44px}
  .brand{font-size:17px}
  .hero{margin-bottom:40px}
  .resultbox{padding:28px 22px 20px;border-radius:18px}
  [data-testid="stImage"]{padding:16px;border-radius:18px}
  hr{margin:36px 0}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
}
</style>
""")


# ---------------------------------------------------------------- model + data

@st.cache_resource(show_spinner=False)
def load_recognizer(model_name: str):
    """Download a published checkpoint once and build the real recognizer.

    MultiRecognizer reads the 'architecture' field stored in the checkpoint,
    so the same class covers all three saved models.
    """
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_MODELS_REPO,
                           filename=MODELS[model_name]["file"])
    return MultiRecognizer(path)


@st.cache_data(show_spinner=False)
def load_examples(model_name: str) -> list[dict]:
    """Selected demo images with their recorded labels.

    Prefers the expanded gallery published on Hugging Face (380 images the
    Transformer matched exactly during screening). Falls back to the 20 images
    bundled in the repo if that download is unavailable.
    """
    key = MODELS[model_name]["key"] if model_name in MODELS else "transformer"
    expanded = _load_expanded(key)
    return expanded if expanded else _load_bundled()


def _load_expanded(folder: str) -> list[dict]:
    """<folder>/ inside Expanded_Khmer_OCR_Demo.zip, with its answers.csv."""
    import csv
    try:
        from huggingface_hub import hf_hub_download
        archive = hf_hub_download(repo_id=HF_DEMO_REPO, repo_type="dataset",
                                  filename=HF_DEMO_FILE)
    except Exception as exc:
        print(f"[khmer-ocr] expanded gallery unavailable: {exc!r}", flush=True)
        return []

    target = DEMO_DIR / "expanded"
    answers = target / folder / "answers.csv"
    if not answers.exists():
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [m for m in zf.namelist() if m.startswith(f"{folder}/")]
                zf.extractall(target, members=members)
        except Exception as exc:
            print(f"[khmer-ocr] could not unpack gallery: {exc!r}", flush=True)
            return []
    if not answers.exists():
        return []

    out = []
    with answers.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            img = target / folder / row.get("image", "")
            if img.exists():
                out.append({"path": img, "answer": row.get("answer", ""),
                            "name": img.name})
    return out


@st.cache_data(show_spinner=False)
def load_handwriting(model_name: str) -> list[dict]:
    """Held-out handwritten lines this checkpoint read exactly.

    Same selection rule as the printed gallery, different source: the project's
    held-out 20% of SoyVitou/khmer-handwritten-dataset-4.2k, which no published
    checkpoint trained on. Empty until screen_handwriting.py has been run, and
    the section is simply not shown in that case.
    """
    import csv
    key = MODELS[model_name]["key"] if model_name in MODELS else "transformer"
    answers = HAND_DIR / key / "answers.csv"
    if not answers.exists():
        return []
    out = []
    with answers.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            img = HAND_DIR / "images" / row.get("image", "")
            if img.exists():
                out.append({"path": img, "answer": row.get("answer", ""),
                            "name": img.name})
    return out


def _load_bundled() -> list[dict]:
    """The 20-image set committed in the repo."""
    if not DEMO_ZIP.exists():
        return []
    if not (DEMO_DIR / "demo").exists():
        with zipfile.ZipFile(DEMO_ZIP) as zf:
            zf.extractall(DEMO_DIR)
    manifest = DEMO_DIR / "demo" / "manifest.json"
    if not manifest.exists():
        return []
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        img = DEMO_DIR / "demo" / row.get("image", "")
        if img.exists():
            out.append({"path": img, "answer": row.get("answer", ""),
                        "name": img.name})
    return out


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over the reference length."""
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(reference)


def copy_button(text: str) -> None:
    """Real clipboard copy. Streamlit cannot reach the clipboard from Python,
    so this is a minimal component styled to match the other buttons."""
    import html as _html
    import streamlit.components.v1 as components
    payload = _html.escape(json.dumps(text))
    components.html(f"""
<style>
  html,body{{margin:0;padding:0;background:transparent}}
  button{{width:100%;height:42px;border-radius:11px;border:1px solid #E7E7E5;
    background:#FFFFFF;color:#1D1D1F;font-size:14px;font-weight:500;cursor:pointer;
    font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    transition:background .15s ease,border-color .15s ease}}
  button:hover{{background:#F7F7F5;border-color:#D8D8D4}}
  button:focus-visible{{outline:2px solid #1D1D1F;outline-offset:2px}}
</style>
<button id="c" type="button">Copy</button>
<script>
  const t = JSON.parse({payload});
  const b = document.getElementById('c');
  b.addEventListener('click', async () => {{
    try {{
      await navigator.clipboard.writeText(t);
      b.textContent = 'Copied';
    }} catch (e) {{
      const a = document.createElement('textarea');
      a.value = t; document.body.appendChild(a); a.select();
      try {{ document.execCommand('copy'); b.textContent = 'Copied'; }}
      catch (_) {{ b.textContent = 'Press ⌘C'; }}
      a.remove();
    }}
    setTimeout(() => {{ b.textContent = 'Copy'; }}, 1600);
  }});
</script>
""", height=48)


# ---------------------------------------------------------------- page state

state = st.session_state
state.setdefault("image_bytes", None)
state.setdefault("image_name", None)
state.setdefault("is_example", False)
state.setdefault("example_answer", "")
state.setdefault("prediction", None)
state.setdefault("failed", False)
state.setdefault("model", "Transformer")
state.setdefault("compare", {})


def set_image(data: bytes, name: str, *, example: bool = False, answer: str = "") -> None:
    state.image_bytes = data
    state.image_name = name
    state.is_example = example
    state.example_answer = answer
    state.prediction = None
    state.compare = {}
    state.failed = False


def clear_image() -> None:
    set_image(None, None)


# ---------------------------------------------------------------- header, hero

st.html(
    f'<div class="topbar"><span class="brand"><span class="dot"></span>Khmer OCR</span>'
    f'<a class="ghlink" href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub ↗</a></div>')

st.html(
    '<div class="hero">'
    '<h1>Read Khmer text from an image</h1>'
    '<p class="sub">Upload a cropped line and recognise it with any of three '
    'models trained from scratch — Transformer, BiLSTM or BiGRU.</p>'
    '<div class="chips"><span>CTC decoding</span><span>242-character set</span>'
    '<span>Runs on CPU</span></div>'
    '</div>')


# ---------------------------------------------------------------- workspace

if state.image_bytes is None:
    st.html('<p class="droptitle">Upload Khmer text</p>')
    upload = st.file_uploader("Upload Khmer text", type=["png", "jpg", "jpeg"],
                              label_visibility="collapsed")

    if upload is not None:
        set_image(upload.getvalue(), upload.name)
        st.rerun()
else:
    try:
        preview = Image.open(io.BytesIO(state.image_bytes))
        st.image(preview, width="stretch")
    except Exception:
        st.markdown('<p class="filemeta">This file could not be opened as an image.</p>',
                    unsafe_allow_html=True)
    st.markdown(f'<p class="filemeta">{state.image_name}</p>', unsafe_allow_html=True)

    if state.is_example:
        st.markdown('<div style="margin-top:14px"><span class="badge sel">'
                    'Selected demo example</span></div>', unsafe_allow_html=True)

    st.write("")
    st.html('<p class="pickerlabel">Model</p>')
    options = list(MODELS) + [COMPARE_ALL]
    chosen = st.segmented_control(
        "Model", options, default=state.model,
        selection_mode="single", key="model_pick", label_visibility="collapsed")
    if chosen and chosen != state.model:
        state.model = chosen
        state.prediction = None
        state.compare = {}
        state.failed = False
        state.shown = {}
        st.rerun()

    st.write("")
    left, right = st.columns([3, 1])
    with left:
        run = st.button("Recognize text", type="primary", width="stretch")
    with right:
        if st.button("Change image", width="stretch"):
            clear_image()
            st.rerun()

    if run:
        slot = st.empty()
        slot.html(
            '<div style="margin-top:26px">'
            '<div class="loadwrap"><div class="spinner"></div>'
            '<div class="loadtxt"><p class="t">Recognizing Khmer text…</p>'
            f'<p class="d">Analyzing the image with '
            f'{"all three models" if state.model == COMPARE_ALL else "the " + state.model + " model"}.</p></div>'
            '</div><div class="skeleton" style="width:72%"></div>'
            '<div class="skeleton" style="width:46%"></div></div>')
        try:
            if state.model == COMPARE_ALL:
                results = {}
                for name in MODELS:
                    recognizer = load_recognizer(name)
                    started = time.perf_counter()
                    text = recognizer.predict(io.BytesIO(state.image_bytes))
                    results[name] = (text, (time.perf_counter() - started) * 1000)
                state.compare = results
                state.prediction = None
            else:
                recognizer = load_recognizer(state.model)
                started = time.perf_counter()
                text = recognizer.predict(io.BytesIO(state.image_bytes))
                state.prediction = text
                state.compare = {state.model: (text, (time.perf_counter() - started) * 1000)}
            state.failed = False
        except Exception as exc:                          # keep the real cause in logs
            print(f"[khmer-ocr] inference failed: {exc!r}", flush=True)
            state.prediction = None
            state.compare = {}
            state.failed = True
        slot.empty()


# ---------------------------------------------------------------- result

if state.failed:
    st.write("")
    st.error("We couldn't read this image. Try another cropped Khmer text-line "
             "image with clearer text.")

elif state.compare:
    st.write("")
    multi = len(state.compare) > 1
    st.markdown('<p class="sec">Recognized text</p>', unsafe_allow_html=True)

    for name, (text, ms) in state.compare.items():
        shown = text if text.strip() else "No text was recognized in this image."
        st.markdown(
            f'<div class="resultbox{" compact" if multi else ""}">'
            f'<p class="khmer">{shown}</p>'
            f'<div class="resultfoot"><span class="tag">{MODELS[name]["label"]} '
            f'· step {MODELS[name]["step"]}</span>'
            f'<span class="tag mono">{ms:.0f} ms</span></div></div>',
            unsafe_allow_html=True)

    primary = state.model if state.model in state.compare else list(state.compare)[0]
    primary_text = state.compare[primary][0]

    c1, c2, c3 = st.columns([1.1, 1.1, 2.8])
    with c1:
        st.download_button("Download .txt", data=primary_text.encode("utf-8"),
                           file_name=f"{Path(state.image_name or 'result').stem}.txt",
                           mime="text/plain", width="stretch")
    with c2:
        copy_button(primary_text)

    if multi:
        # The comparison table is the point of this mode, so it is always on
        # screen rather than hidden behind a disclosure.
        st.write("")
        st.markdown('<p class="sec">Comparison</p>', unsafe_allow_html=True)

        expected = state.example_answer if state.is_example else ""
        if not expected:
            expected = st.text_input(
                "Expected transcription",
                placeholder="Paste the correct Khmer text to score these predictions",
                key="ref_input", label_visibility="collapsed")

        if expected.strip():
            st.markdown(
                '<p class="reflabel">Reference</p>'
                f'<div class="refbox"><span class="khmer-inline">{expected}</span></div>',
                unsafe_allow_html=True)

        head = ("<th>Model</th><th>Predicted text</th><th>Time</th>"
                + ("<th>CER</th><th>Exact</th>" if expected.strip() else ""))
        rows_html = []
        for name, (text, ms) in state.compare.items():
            cells = (f'<td class="m">{MODELS[name]["label"]}</td>'
                     f'<td class="km-cell">{text if text.strip() else "—"}</td>'
                     f'<td class="num">{ms:.0f} ms</td>')
            if expected.strip():
                match = normalize(expected) == normalize(text)
                cer = char_error_rate(expected, text)
                cells += (f'<td class="num">{cer * 100:.1f}%</td>'
                          f'<td class="num {"ok" if match else "no"}">'
                          f'{"Match" if match else "Different"}</td>')
            rows_html.append(f"<tr>{cells}</tr>")

        st.markdown(
            f'<div class="tablewrap"><table class="cmp"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table></div>'
            '<p class="tablenote">One image is not a benchmark. These figures describe '
            'this single line only, and the timings are first-call CPU measurements.</p>',
            unsafe_allow_html=True)
    else:
        with st.expander("Compare with expected text"):
            expected = st.text_area(
                "Expected transcription",
                value=state.example_answer if state.is_example else "",
                placeholder="Paste the correct Khmer text for this image",
                height=96)
            if expected.strip():
                text = state.compare[primary][0]
                match = normalize(expected) == normalize(text)
                cer = char_error_rate(expected, text)
                st.markdown(
                    '<div class="metricrow"><span class="k">Exact match</span>'
                    f'<span class="v{" good" if match else ""}">'
                    f'{"Match" if match else "Different"}</span></div>'
                    '<div class="metricrow"><span class="k">Character error rate</span>'
                    f'<span class="v">{cer:.2f}</span></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- examples

examples = load_examples(state.model)
handwriting = load_handwriting(state.model)
which = "Transformer" if state.model == COMPARE_ALL else state.model

# Printed and handwriting are different corpora read by the same checkpoint, so
# they are shown as separate galleries rather than one mixed grid. The
# handwriting section disappears when its folder has not been built.
SECTIONS = [s for s in (
    ("Printed", examples,
     f"{len(examples)} images the {which} checkpoint matched exactly during "
     f"screening of 500 validation lines"),
    ("Handwriting", handwriting,
     f"{len(handwriting)} lines the {which} checkpoint read exactly, from the 840 "
     f"held-out images of the handwriting corpus it never trained on"),
) if s[1]]

if SECTIONS:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec">Try an example</p>', unsafe_allow_html=True)

    names = [name for name, _, _ in SECTIONS]
    state.setdefault("section", names[0])
    if state.section not in names:
        state.section = names[0]
    if len(names) > 1:
        picked = st.segmented_control(
            "Example type", names, default=state.section,
            selection_mode="single", key="gallery_pick",
            label_visibility="collapsed")
        if picked and picked != state.section:
            state.section = picked
            st.rerun()
        st.write("")

    name, items, caption = next(s for s in SECTIONS if s[0] == state.section)
    st.markdown(f'<p class="sec-sub">{caption}</p>', unsafe_allow_html=True)

    per_row, page = 4, 12
    # one page counter per section, so paging through one does not expand the other
    if not isinstance(state.get("shown"), dict):
        state.shown = {}
    shown = min(state.shown.get(name, page), len(items))
    picks = items[:shown]

    for start in range(0, len(picks), per_row):
        row = picks[start:start + per_row]
        cols = st.columns(per_row)
        for col, item in zip(cols, row):
            index = items.index(item) + 1
            with col, st.container(key=f"tile{name}{index}"):
                st.image(str(item["path"]), width="content")
                if st.button(f"Example {index:02d}", key=f"ex{name}{index}",
                             width="stretch"):
                    set_image(item["path"].read_bytes(), item["name"],
                              example=True, answer=item["answer"])
                    st.rerun()
        # keep the trailing columns of a short last row empty, not stretched
        for col in cols[len(row):]:
            with col:
                st.empty()

    st.write("")
    st.html(f'<p class="galcount">Showing {shown} of {len(items)}</p>')
    want_more = shown < len(items)
    want_less = shown > page
    remaining = len(items) - shown

    def more_button():
        if st.button(f"Show {min(page, remaining)} more", key=f"more{name}"):
            state.shown[name] = shown + page
            st.rerun()

    def less_button():
        if st.button("Collapse", key=f"less{name}"):
            state.shown[name] = page
            st.rerun()

    # No spacer columns: an empty column still takes half the row and pushes a
    # lone button off-centre. Each button is centred by CSS instead.
    if want_more and want_less:
        left, right = st.columns(2)
        with left:
            more_button()
        with right:
            less_button()
    elif want_more:
        more_button()
    elif want_less:
        less_button()

    note = ('Each model has its own set of images it read correctly during '
            'screening, so the gallery changes with the model. Counts differ '
            'because the screening selected successes; they are not a fair '
            'benchmark and say nothing about accuracy on new images.')
    if state.section == "Handwriting":
        note = ('The source corpus is labelled handwriting but is mixed: it also '
                'holds printed lines, so a high match count here is not a '
                'handwriting result. Only lines containing Khmer are kept, and '
                'these are selected successes -- not a benchmark.')
    st.markdown(f'<p class="sec-sub" style="margin-top:20px">{note}</p>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- notes, footer

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="note"><h4>Designed for text-line recognition.</h4>'
    '<p>For best results, upload one cropped line of Khmer text. Full-page '
    'document recognition is not currently supported.</p></div>',
    unsafe_allow_html=True)

st.write("")
st.markdown(
    '<p class="sec" style="margin-top:28px">About the models</p>'
    '<p class="about">Three saved architectures, all CTC-decoded: '
    '<b>Transformer</b> (step 87,000), <b>BiLSTM</b> (58,000), '
    '<b>BiGRU</b> (8,500)<br>'
    'Runs on CPU<br>'
    f'Checkpoints hosted on Hugging Face · <b>{HF_MODELS_REPO}</b><br>'
    'Saved steps are not comparable as training effort, and these checkpoints '
    'were exported without optimizer state.</p>',
    unsafe_allow_html=True)

st.markdown('<p class="foot">Khmer OCR · Transformer OCR Demo</p>',
            unsafe_allow_html=True)
