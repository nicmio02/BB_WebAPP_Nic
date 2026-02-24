"""
smart_parser_two_pass.py
------------------------
Two-pass PDF extraction using Gemini:
  Phase 1 (Scout): Discover document structure (samples + layout)
  Phase 2 (Miner): Extract parameter/value/unit data per chunk

Returns a long-format DataFrame with columns:
  sample_id, parameter, value, unit
"""

import os
import time
from typing import List

import fitz  # PyMuPDF
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig

# Global client placeholder
client = None


# --- UTILS ---
def generate_with_retry(model_name: str, contents: list, config: GenerateContentConfig,
                        retries: int = 5, base_delay: int = 5):
    """Wraps Gemini calls with exponential backoff for 429 errors."""
    global client
    if not client:
        raise ValueError("Gemini Client not initialized. Pass api_key to process_generic_report.")

    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = base_delay * (2 ** attempt)
                print(f"   ⚠️ Quota hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded for Gemini API.")


# --- DATA STRUCTURES ---
class SampleLocation(BaseModel):
    sample_id: str = Field(description="The name of the sample (e.g., 'MV27', 'Soil_1').")
    layout_type: str = Field(description="Either 'MATRIX' (columns) or 'SEQUENTIAL' (pages).")
    location_key: str = Field(
        description="If MATRIX: the column index/header. If SEQUENTIAL: the page number or section title.")


class DocumentStructure(BaseModel):
    samples: List[SampleLocation] = Field(description="List of all samples found and their locations.")
    notes: str = Field(description="Any specific notes on how to interpret the tables.")


class ExtractionResult(BaseModel):
    sample_id: str = Field(description="The Sample ID this value belongs to.")
    parameter: str = Field(description="The name of the chemical/physical parameter.")
    value: str = Field(description="The numeric value or text result.")
    unit: str = Field(description="The unit of measurement.")


class PageExtraction(BaseModel):
    results: List[ExtractionResult]


# --- HELPER FUNCTIONS ---

def get_pdf_text_layout(file_path: str) -> List[str]:
    """Reads PDF and keeps page boundaries clear."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text", sort=True)
        header = f"--- PAGE {i + 1} ---\n"
        pages.append(header + text)
    return pages


# --- PHASE 1: THE SCOUT ---
def discover_structure(all_pages: List[str]) -> DocumentStructure:
    print("🗺️  Phase 1: Scouting Document Structure...")

    full_context = "\n".join(all_pages)

    prompt = """
    You are a document layout analyst. Your job is to create a "Map" of this scientific report.

    Task:
    1. Identify all unique **Samples** being analyzed. Look for "Monsteromschrijving", "Sample ID", or "Project Name" sections.
    2. Determine if the results are presented in a **MATRIX** (columns represent samples, rows represent parameters) or **SEQUENTIAL** (one sample per page/section).
    3. If MATRIX: Identify which Column Number or Header maps to which Sample ID.
    4. If SEQUENTIAL: Identify which identifier signifies the start of that sample.

    Return the list of samples and their location keys.
    """

    try:
        response = generate_with_retry(
            model_name="gemini-2.5-flash-lite",
            contents=[prompt, full_context],
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DocumentStructure,
                temperature=0.0,
            ),
        )
        structure = response.parsed
        print(f"   -> Discovered {len(structure.samples)} samples.")
        return structure
    except Exception as e:
        print(f"   -> Discovery Failed: {e}")
        return DocumentStructure(samples=[], notes="Failed to detect.")


# --- PHASE 2: THE MINER ---
def extract_data_with_map(page_text: str, structure: DocumentStructure,
                          chunk_index: int) -> List[ExtractionResult]:
    """Extract data using the discovered map as context."""

    structure_json = structure.model_dump_json()

    prompt = f"""
    You are a Scientific Data Extractor. 

    ### THE MAP (Use this to understand the layout)
    {structure_json}

    ### INSTRUCTIONS
    1. Analyze the text below.
    2. Extract chemical analysis parameters, values, and units.
    3. **CRITICAL**: Use the "Map" above to assign the correct **Sample ID** to every value.
       - If the Map says "MATRIX" and Column 1 is "MV27", then the first value in a data row belongs to "MV27".
       - If the Map says "SEQUENTIAL" and you see a header matching the location key, assign data to that sample.
    4. Ignore page numbers, footer info, and disclaimer text.
    5. Handle "less than" signs (e.g., "<0.1") as part of the value.

    ### INPUT TEXT
    {page_text}
    """

    try:
        response = generate_with_retry(
            model_name="gemini-2.5-flash-lite",
            contents=prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PageExtraction,
                temperature=0.0,
            ),
        )
        return response.parsed.results
    except Exception as e:
        print(f"   -> Extraction Error on chunk {chunk_index}: {e}")
        return []


# --- MAIN PIPELINE ---

def process_generic_report(file_path: str, api_key: str | None = None) -> pd.DataFrame:
    """
    Main entry point. Extracts a long-format DataFrame from a soil report PDF.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.
    api_key : str | None
        Gemini API key. Falls back to GEMINI_KEY env var if not provided.

    Returns
    -------
    pd.DataFrame
        Long-format with columns: sample_id, parameter, value, unit.
    """
    global client

    # Initialize Client
    if api_key:
        client = genai.Client(api_key=api_key)
    elif not client:
        env_key = os.getenv("GEMINI_KEY")
        if env_key:
            client = genai.Client(api_key=env_key)
        else:
            raise ValueError("No GEMINI_KEY provided. Pass api_key argument or set env var.")

    # 1. Read File
    pages = get_pdf_text_layout(file_path)

    # 2. Scout Structure
    structure = discover_structure(pages)

    if not structure.samples:
        print("⚠️ No samples detected. Manual review required.")
        return pd.DataFrame(columns=["sample_id", "parameter", "value", "unit"])

    # 3. Mine Data (chunking by 3 pages)
    all_results = []
    chunk_size = 3
    print(f"⛏️  Phase 2: Mining Data ({len(pages)} pages)...")

    chunk_index = 0
    for i in range(0, len(pages), chunk_size):
        chunk = "\n".join(pages[i:i + chunk_size])
        chunk_index += 1
        print(f"   -> Processing pages {i + 1} to {min(i + chunk_size, len(pages))}...")

        extracted = extract_data_with_map(chunk, structure, chunk_index)
        all_results.extend(extracted)
        time.sleep(1)  # Rate limiting

    # 4. Build DataFrame
    df = pd.DataFrame([vars(r) for r in all_results])

    if not df.empty:
        df.drop_duplicates(subset=["sample_id", "parameter", "unit"], keep="last", inplace=True)

    print(f"\n✅ Extraction Complete — {len(df)} rows.")
    return df
