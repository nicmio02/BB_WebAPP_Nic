# Local Testing Guide — Soil Analysis Pipeline

## 1. Project Structure

Place the updated files into your existing project like this:

```
bagger-tool/
├── Home.py                          
├── app_common.py                    
├── auth_config.py                   
├── FeatureList.py                   
├── matcher.py                       
├── smart_parser_two_pass.py         
├── baggerTool_v7.db                 
├── .streamlit/
│   ├── config.toml                  
│   └── secrets.toml                 
├── db/
│   ├── __init__.py                  
│   └── samples_store.py             
├── cbc/
│   ├── __init__.py                  
│   └── cbc_core_advanced.py         
├── visuals/
│   ├── __init__.py                  
│   └── visuals_advanced.py          
├── pages/
│   ├── Analyse.py                   
│   └── Parameter_Configuratie.py    
└── user_profile_logos/              
```

> **Keep the old files** (cbc_core.py, visuals.py) around until you've
> confirmed everything works. You can rename them with a `.bak` extension.


## 2. Environment Setup

```bash
# Create/activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn plotly
pip install google-genai pydantic PyMuPDF
pip install streamlit-authenticator  # for auth_config

# Verify key packages
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
python -c "import fitz; print(f'PyMuPDF {fitz.version}')"
python -c "from google import genai; print('google-genai OK')"
```


## 3. Secrets Configuration

Create/update `.streamlit/secrets.toml`:

```toml
GEMINI_KEY = "your-gemini-api-key-here"
```

Or set it as an environment variable:

```bash
export GEMINI_KEY="your-gemini-api-key-here"
```


## 4. Quick Smoke Test (No Browser Needed)

Run this script to test the pipeline end-to-end without Streamlit:

```bash
python test_pipeline.py your_report.pdf
```

Create `test_pipeline.py` in the project root (see the file included in the download).


## 5. Run the Streamlit App

```bash
streamlit run Home.py
```

### Test checklist:

- [ ] **Login** — Does the auth page render?
- [ ] **Upload a PDF** — Does extraction start? Check terminal for Phase 1/2 logs
- [ ] **Matching progress** — Do you see the progress bar cycling through samples?
- [ ] **Success message** — "Stored N matched samples" appears?
- [ ] **Navigate to Analyse** — Do samples appear in the multiselect?
- [ ] **Select samples** — Do visuals render (bar chart, spider, diverging bars)?
- [ ] **Re-upload same PDF** — Does it show "Scored results already exist" message?
- [ ] **Parameter Configuratie** — Does the page load without errors?


## 6. Debugging Common Issues

### "ModuleNotFoundError: No module named 'FeatureList'"
→ `FeatureList.py` must be in the project root (same directory as `Home.py`).

### "ModuleNotFoundError: No module named 'db'"
→ Create `db/__init__.py` (empty file).

### "ModuleNotFoundError: No module named 'cbc'"
→ Create `cbc/__init__.py` (empty file).

### "GEMINI_KEY is missing"
→ Add it to `.streamlit/secrets.toml` or set the environment variable.

### "Lokale regeldatabase 'baggerTool_v7.db' niet gevonden"
→ `baggerTool_v7.db` must be in the project root (same directory as `Home.py`).

### Analyse page shows "Nog geen opgeslagen monsters"
→ You need to upload+process a PDF first on the Home page. Old data from
  `extracted_samples` won't appear — only data in the new `scored_samples`
  table is used.

### Matcher returns all -1 values
→ Check terminal for Gemini API errors. May be a quota or key issue.
  Also verify the extraction produced sensible parameter names by checking
  the raw extraction in the terminal output.

### Charts don't render / matplotlib errors
→ Ensure `matplotlib` backend is `Agg`. The updated visuals file sets this,
  but if you have a `matplotlibrc` file overriding it, that could conflict.


## 7. Database Migration Note

The new `samples_store.py` creates a `scored_samples` table automatically
on first connection. Your existing `samples.db` will gain this table
alongside the existing `extracted_samples` and `parameter_mappings` tables.

**No data is deleted.** But the Analyse page now reads from `scored_samples`,
so previously uploaded PDFs need to be re-uploaded to go through the new
AI matching pipeline.

If you want to back up your existing database first:

```bash
cp samples.db samples.db.backup
```
