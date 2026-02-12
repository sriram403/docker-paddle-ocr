import streamlit as st
import fitz 
import json
import os
import gc # <--- NEW: Garbage Collector
import re
import nltk
import openai
import pandas as pd
from collections import defaultdict
import base64
import logging
import httpx
import unicodedata
import concurrent.futures
import cv2
import numpy as np
import paddle
try:
    # 1. Force Dynamic Mode (Fixes the int(Tensor) error)
    paddle.disable_static()
    # 2. Try to limit GPU memory fraction if possible (Optional, good for stability)
    # paddle.device.set_device('gpu:0') 
except Exception:
    pass

try:
    from paddleocr import PPStructureV3, PaddleOCRVL
except ImportError:
    pass

@st.cache_resource
def download_nltk_data():
    packages = ['punkt', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

download_nltk_data()

@st.cache_resource
def load_paddle_model(model_type="vl"):
    try:
        # Force garbage collection before loading
        gc.collect()
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()

        if model_type == "vl":
            return PaddleOCRVL()
        else:
            return PPStructureV3(
                layout_detection_model_name="PP-DocLayout-L", 
                text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                device="gpu", # Change to "cpu" if your computer keeps crashing
                use_table_recognition=True,    
                use_doc_orientation_classify=False,
                use_region_detection=True,
                use_doc_unwarping=False
            )
    except Exception as e:
        st.error(f"Failed to load PaddleOCR: {e}")
        return None

class TableExtractor:
    def __init__(self, openai_api_key, model="gpt-4o", log_file="table_extraction_debug.log"):
        self.api_key = openai_api_key
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        if self.api_key:
            custom_http_client = httpx.Client(
                http2=False,
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            self.client = openai.OpenAI(
                api_key=self.api_key,
                http_client=custom_http_client,
                max_retries=2
            )
        else:
            self.client = None

    def _convert_page_to_image(self, page, dpi=200, crop_rect=None):
        if crop_rect:
            # Add small padding to crop
            crop_rect = crop_rect + (-5, -5, 5, 5)
            # Ensure crop is within page bounds
            crop_rect = crop_rect & page.rect
            pix = page.get_pixmap(dpi=dpi, clip=crop_rect)
        else:
            pix = page.get_pixmap(dpi=dpi)
            
        img_bytes = pix.tobytes("jpeg", jpg_quality=85)
        return base64.b64encode(img_bytes).decode('utf-8')

    def _check_vector_graphics(self, page):
        h_lines, v_lines = 0, 0
        for path in page.get_drawings():
            for item in path["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) < 2: h_lines += 1
                    elif abs(p1.x - p2.x) < 2: v_lines += 1
                elif item[0] == "re":
                    h_lines += 2; v_lines += 2
        if h_lines > 5 and v_lines > 2: return 50, f"Strong vector graphics (H: {h_lines}, V: {v_lines})"
        elif h_lines > 10 or v_lines > 5: return 25, f"Moderate vector graphics (H: {h_lines}, V: {v_lines})"
        return 0, "No vector graphics found"

    def _check_keywords(self, page):
        text = page.get_text().lower()
        if re.search(r'table of contents|contents', text): return 0, "TOC detected"
        matches = []
        if re.search(r'\btable\s+([ivx\d]+|[\da-z])\b', text): matches.append("Table X")
        if re.search(r'\bannex\s+([ivx\d]+|[\da-z])\b', text): matches.append("Annex X")
        if matches: return 30, f"Keywords: {', '.join(matches)}"
        return 0, "No keywords"

    def _check_header_heuristics(self, page):
        max_caps = 0
        for block in page.get_text("dict")["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    caps = 0
                    words = []
                    for span in line['spans']: words.extend(span['text'].split())
                    for w in words:
                        if len(w)>2 and w.isupper() and re.match(r'^[A-Z\(\)]+$', w): caps += 1
                    if caps > max_caps: max_caps = caps
        if max_caps >= 4: return 30, f"Strong Headers ({max_caps} CAPS words)"
        elif max_caps >= 2: return 20, f"Moderate Headers ({max_caps} CAPS words)"
        return 0, "No headers"

    def _check_columnar_text(self, page):
        cols = defaultdict(int)
        words = page.get_text("words")
        if not words: return 0, "No text"
        for w in words: cols[round(w[0]/10)] += 1
        sig = len([k for k, c in cols.items() if c > 3])
        if sig >= 5: return 30, f"Strong Columns ({sig})"
        elif sig >= 3: return 20, f"Moderate Columns ({sig})"
        return 0, "No columns"

    def _check_explicit_column_headers(self, page):
        hits = len(re.findall(r'\bcolumn\s+\d+\b', page.get_text(), re.IGNORECASE))
        if hits >= 2: return 200, f"Override: {hits} 'Column X' found"
        elif hits == 1: return 120, "Strong: 'Column X' found"
        return 0, "No explicit headers"

    def _check_fitz_table_finder(self, page):
        try:
            if page.find_tables(strategy="lines").tables: return 30, "Fitz Lines detected"
            if page.find_tables(strategy="text").tables: return 15, "Fitz Text detected"
        except: pass
        return 0, "Fitz found nothing"

    def score_page(self, page, page_num):
        ui_hits = []
        score, reason = self._check_explicit_column_headers(page)
        if score > 0: ui_hits.append("Explicit Headers")
        
        if score < 200:
            checks = [self._check_fitz_table_finder, self._check_vector_graphics, 
                      self._check_keywords, self._check_header_heuristics, self._check_columnar_text]
            for func in checks:
                s, r = func(page)
                score += s
                if s > 0:
                    clean_name = func.__name__.replace("_check_", "").replace("_", " ").title()
                    ui_hits.append(f"{clean_name} ({s})")

        summary_reason = ", ".join(ui_hits) if ui_hits else "No clear table signals"
        return score, summary_reason

    def _extract_page_sync(self, page_data):
        # Unpack tuple: now expects (page, page_num, optional_rect)
        if len(page_data) == 3:
            page, page_num, crop_rect = page_data
        else:
            page, page_num = page_data
            crop_rect = None

        if not self.client: return page_num, []
        
        try:
            b64_image = self._convert_page_to_image(page, crop_rect=crop_rect)
            system_prompt = """
            You are a data extraction engine. Extract data from images into machine-readable JSON.
            OUTPUT FORMAT: { "tables": [ { "title": "string", "headers": ["s1", "s2"], "rows": [ ["s1", "s2"], ... ] } ] }
            """
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract table data exactly as seen."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]}
                ],
                temperature=0.0
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            processed = []
            
            for i, t in enumerate(data.get("tables", [])):
                df = pd.DataFrame(t.get("rows", []))
                headers = t.get("headers", [])
                
                if headers:
                    new_headers = []
                    seen_counts = defaultdict(int)
                    for header in headers:
                        current_count = seen_counts[header]
                        new_headers.append(f"{header}.{current_count}" if current_count > 0 else header)
                        seen_counts[header] += 1
                    
                    if len(new_headers) == df.shape[1]: df.columns = new_headers
                    elif len(new_headers) > df.shape[1]: df.columns = new_headers[:df.shape[1]]
                    else: df.columns = new_headers + [f"Col_{k}" for k in range(df.shape[1]-len(new_headers))]
                
                processed.append({
                    "page_number": page_num,
                    "table_number": i + 1,
                    "title": t.get("title", f"Table on Page {page_num}"),
                    "html": df.to_html(index=False, border=1, classes="dataframe"),
                    "df": df,
                    "b64_img_ref": b64_image
                })
            return page_num, processed
        except Exception:
            return page_num, []

    def process_pages_concurrently(self, doc, pages_to_process, status_callback):
        results = {}
        tasks = []
        for p_num in pages_to_process:
            tasks.append((doc[p_num - 1], p_num))
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {executor.submit(self._extract_page_sync, task): task[1] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_page):
                try:
                    p_num, tables = future.result()
                    if tables: results[p_num] = tables
                except Exception: pass
                status_callback()
        return results

class DocumentProcessor:
    def __init__(self):
        self.openai_api_key = None
        log_file = "doc_processor_debug.log"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)

        # --- NEW LABEL CONFIGURATION (Derived from CSV) ---
        self.LABEL_SPECS = {
            "Definitions": {
                "Purpose": "Identify clauses that define terms used by the regulation; source of canonical term meanings.",
                "Regex_Patterns": r"(?i)\bdefinitions?\b|\bterms\s+and\s+definitions\b|\bterminology\b|\"[^\"]+\"\s+means\b|\brefers\s+to\b|\bdenotes\b"
            },
            "Applicability/Scope": {
                "Purpose": "Determine entities, products, and conditions covered by the regulation.",
                "Regex_Patterns": r"(?i)\bscope\b|\bapplicability\b|\bfield\s+of\s+application\b|\b(this\s+(regulation|standard)\s+appl(ies|y))\b|\bshall\s+apply\s+to\b|\bcovers?\b|\bwithin\s+the\s+scope\b"
            },
            "Alternative/Equivalency/References": {
                "Purpose": "Detect cross-references, equivalencies to other clauses/annexes/standards, and permissive alternates.",
                "Regex_Patterns": r"(?i)\balternative(s)?\b|\bequivalen(ce|t)\b|\bmutatis\s+mutandis\b|\b(refer|reference|see)\s+(to|clause|annex|appendix)\b|Annex\s+\d+|Appendix\s+[A-Z]|\bClause\s+\d+(\.\d+)*"
            },
            "Effective Dates/Commencement/Transitional": {
                "Purpose": "Capture commencement dates, transitional provisions, and sunset rules.",
                "Regex_Patterns": r"(?i)\b(effective|commencement|entry\s+into\s+force|date\s+of\s+application)\b|\btransitional\s+provisions?\b|\bsunset\b|\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
            },
            "Type Approval / Approval Evidence / Marking": {
                "Purpose": "Identify homologation steps, approval marks/numbers, conformity certificates.",
                "Regex_Patterns": r"(?i)\btype\s+approval\b|\bapproval\s+(mark|number|certificate)\b|\bconformity\s+certificate\b|\bCoC\b|\bE\s*-\s*mark\b|\bmark(ing)?\b(?!\s*up\b)"
            },
            # --- NEW LABEL 1: HMI ---
            "HMI (Human-Machine Interface)": {
                "Purpose": "Identify user interaction elements: displays, tell-tales, warnings (visual/audible), controls, and driver alerts.",
                "Regex_Patterns": r"(?i)\bHMI\b|\b(visual|audible|tactile|optical|acoustic)\s+(warning|signal|alert|indication)\b|\btell[- ]?tale\b|\b(malfunction|status)\s+indicator\b|\bdashboard\b|\bdisplay\b"
            },
            # --- NEW LABEL 2: HANDBOOK ---
            "Handbook / User Manual": {
                "Purpose": "Identify content intended for end users/drivers: owner's manuals, interpreting indications, and safety responses.",
                "Regex_Patterns": r"(?i)\b(owner['’]?s|user)\s+(manual|handbook|guide)\b|\binstructions?\s+(to|for)\s+(the\s+)?(driver|user)\b|\bdescribe\s+in\s+the\s+manual\b"
            },
            "Documentation": {
                "Purpose": "Find documentary requirements: technical file, test/inspection reports, retention rules.",
                "Regex_Patterns": r"(?i)\bdocumentation\b|\brecord(s|keeping)\b|\bretain(ed)?\s+for\s+\d+\s+(year|month)s?\b|\btechnical\s+file\b|\btest\s+report\b|\binspection\s+report\b"
            },
            "COP (Conformity of Production)": {
                "Purpose": "Detect ongoing production conformity obligations after approval.",
                "Regex_Patterns": r"(?i)\bconformity\s+of\s+production\b|\bCoP\b|\bproduction\s+conformity\b|\bseries\s+production\b"
            },
            "Enforcement / Penalties": {
                "Purpose": "Identify sanctions and actions for non-compliance.",
                "Regex_Patterns": r"(?i)\b(enforcement|penalt(y|ies)|sanction(s)?)\b|\bnon[-\s]?compliance\b|\bshall\s+be\s+liable\b|\bwithdraw(al)?\b|\bsuspension\b|\brevocation\b"
            },
            "Test Conditions": {
                "Purpose": "Capture environmental/setup parameters for tests (temp, voltage, speed, load, humidity).",
                "Regex_Patterns": r"(?i)\btest\s+conditions?\b|\bambient\b|\bpre[- ]conditioning\b|(temperature|voltage|speed|load|humidity).{0,40}\b(\d+(\.\d+)?)(\s*(°C|V|km/h|N|%))"
            },
            "Test Methods": {
                "Purpose": "Identify procedural steps for executing tests (measurement procedures/protocols).",
                "Regex_Patterns": r"(?i)\btest\s+method(s)?\b|\bmeasurement\s+method\b|\bprocedure\b|\bprotocol\b|\btesting\s+shall\s+be\s+carried\s+out\b"
            },
            "Test Scenarios": {
                "Purpose": "Identify test cases or specific scenario setups (e.g., driving cycles, manoeuvres).",
                "Regex_Patterns": r"(?i)\b(test\s+)?scenario(s)?\b|\btest\s+case(s)?\b|\bdriving\s+cycle\b|\bmanoeuv(re|er)\b|\b(e\.g\.|for example)\b"
            },
            "Exemptions & Alternative Procedures": {
                "Purpose": "Detect exceptions, waivers, and conditional relief statements.",
                "Regex_Patterns": r"(?i)\bexempt(ion|ed|s)?\b|\bwaiver\b|\bnot\s+required\s+if\b|\bin\s+lieu\s+of\b|\balternative\s+procedure(s)?\b"
            },
            "Notes/Preamble/Editorial": {
                "Purpose": "Mark non-normative context to avoid false positives for requirements.",
                "Regex_Patterns": r"(?i)^\s*(note|notes)\s*[:\-]|\bpreamble\b|\beditorial\b|\bforeword\b|\bintroduction\b"
            },
            "Out of Scope": {
                "Purpose": "Explicit exclusions from applicability.",
                "Regex_Patterns": r"(?i)\bout\s+of\s+scope\b|\bdoes\s+not\s+apply\s+to\b|\bexcluded\s+from\s+the\s+scope\b|\bexclusion(s)?\b"
            }
        }

    def _get_symbol_map(self):
        return {
            'a': 'α', 'b': 'β', 'g': 'γ', 'd': 'δ', 'e': 'ε', 'z': 'ζ', 'h': 'η', 'q': 'θ',
            'i': 'ι', 'k': 'κ', 'l': 'λ', 'm': 'μ', 'n': 'ν', 'x': 'ξ', 'o': 'ο', 'p': 'π',
            'r': 'ρ', 's': 'σ', 't': 'τ', 'u': 'υ', 'f': 'φ', 'c': 'χ', 'y': 'ψ', 'w': 'ω',
            'A': 'Α', 'B': 'Β', 'G': 'Γ', 'D': 'Δ', 'E': 'Ε', 'Z': 'Ζ', 'H': 'Η', 'Q': 'Θ',
            'I': 'Ι', 'K': 'Κ', 'L': 'Λ', 'M': 'Μ', 'N': 'Ν', 'X': 'Ξ', 'O': 'Ο', 'P': 'Π',
            'R': 'Ρ', 'S': 'Σ', 'T': 'Τ', 'U': 'Υ', 'F': 'Φ', 'C': 'Χ', 'Y': 'Ψ', 'W': 'Ω',
            '°': '°',
        }

    def _calculate_iou(self, boxA, boxB):
        if not boxA or not boxB: return 0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = float(boxAArea + boxBArea - interArea)
        return interArea / unionArea if unionArea != 0 else 0

    def _get_sorting_key(self, item):
        label = item.get('label', '').lower()
        if 'header' in label: return -1000
        if 'footer' in label: return 10000
        if 'order' in item and item['order'] is not None:
            return int(item['order'])
        bbox = item.get('bbox', [0, 0, 0, 0])
        return int(bbox[1])

    def hybrid_extract_and_structure(self, doc, num_pages, paddle_model, model_type="v3", progress_callback=None):
        all_chunks = []
        parent_stack = [(0, "ROOT")] 
        current_annex_context = "NIL"
        chunk_counter = 1
        in_toc_mode = False
        toc_ids_seen = set()
        
        metrics = {
            "total_words": 0, "kept_words": 0, "dropped_words": 0, 
            "debug_images": {}, "detected_tables": defaultdict(list)
        }
        
        current_chunk = {
            'chunk_id': 1, 'clause_id': "NIL", 'parent_id': "ROOT", 'level': 0, 'Title': "NIL",
            'appendix/annex': "NIL", 'content_verbatim': "", 'source_page': "",
            'content_words': [] 
        }
        
        patterns = self.get_compliance_patterns()
        patterns['content_start_heading'] = r'^\s*(INTRODUCTION|FOREWORD|PREAMBLE|PURPOSE|SCOPE)(?![a-z])'
        
        regex_map = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
        
        heading_priority = [
            'toc_start_heading', 'preamble_heading', 'content_start_heading', 'appendix_heading', 'chapter_heading', 
            'section_general_heading', 'cfr_part_heading', 'cfr_section_heading', 'fmvss_paragraph', 
            'article_heading', 'roman_upper_heading', 'multi_level_heading', 'numeric_paren_heading', 
            'alpha_paren_heading'
        ]
        
        month_pattern = re.compile(r'(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\b')

        limit = min(len(doc), num_pages)
        
        for p_idx in range(limit):
            if progress_callback:
                progress_callback((p_idx + 1) / limit)

            p_num = p_idx + 1
            page = doc[p_idx]
            
            # --- FIX: Calculate Total Words from Raw Page (True Baseline) ---
            try:
                raw_words = page.get_text("words")
                metrics['total_words'] += len(raw_words)
            except: pass
            # ----------------------------------------------------------------

            zoom = 2.0 if model_type == "vl" else 1.5 
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            
            if pix.n < 3: pix = fitz.Pixmap(fitz.csRGB, pix)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 4: img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else: img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            img_bgr = np.ascontiguousarray(img_bgr)

            try:
                layout_results = paddle_model.predict(img_bgr)
            except Exception as e:
                st.error(f"Page {p_num} Error: Vision Model failed ({e}). Try switching to 'v3'.")
                continue

            all_blocks = self._parse_paddle_output(layout_results, model_type)
            valid_blocks = [b for b in all_blocks if b['label'].lower() not in ['header', 'footer']]
            valid_blocks.sort(key=self._get_sorting_key)

            viz_image = img_bgr.copy()
            scale_x = page.rect.width / img_bgr.shape[1]
            scale_y = page.rect.height / img_bgr.shape[0]

            extracted_lines = [] 
            table_counter = 1
            
            if str(p_num) not in current_chunk['source_page']:
                sep = ", " if current_chunk['source_page'] else ""
                current_chunk['source_page'] += f"{sep}{p_num}"

            for idx, block in enumerate(valid_blocks):
                label = block['label'].lower()
                bbox = block['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                color = (0, 0, 255) if 'table' in label else (0, 255, 0)
                if 'title' in label: color = (255, 0, 0)
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                
                pdf_rect = fitz.Rect(bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y)
                
                if 'table' in label:
                    metrics['detected_tables'][p_num].append(pdf_rect)
                    line_text = f"<Table_{table_counter}_Pg{p_num}>"
                    extracted_lines.append((line_text, label, [], []))
                    table_counter += 1
                else:
                    block_dict = page.get_text("dict", clip=pdf_rect)
                    for b in block_dict.get("blocks", []):
                        if "lines" not in b: continue
                        for l in b["lines"]:
                            line_text_parts = []
                            line_words = []
                            line_rich = []
                            
                            for s in l["spans"]:
                                txt = self._clean_text(s["text"], s["font"])
                                if txt.strip():
                                    line_text_parts.append(txt)
                                    is_bold = self._is_span_bold(s)
                                    line_rich.append({"text": txt, "is_bold": is_bold})
                                    line_words.append({
                                        "page": p_num, "text": txt, "bbox": [round(x) for x in s["bbox"]]
                                    })
                            
                            full_line = " ".join(line_text_parts).strip()
                            if full_line:
                                extracted_lines.append((full_line, label, line_words, line_rich))

            success, buffer = cv2.imencode(".jpg", viz_image)
            if success: metrics['debug_images'][p_num] = buffer.tobytes()

            merged_lines = []
            k = 0
            while k < len(extracted_lines):
                curr = extracted_lines[k]
                curr_text = curr[0]
                
                is_dangling = False
                if len(curr_text) < 20:
                    if re.match(r'^[\d\.]+$', curr_text) or re.match(r'^(I|V|X|CHAPTER|SECTION|PART|ARTICLE)\s*[\dA-Z\.]*$', curr_text, re.IGNORECASE):
                        is_dangling = True
                
                if is_dangling and k + 1 < len(extracted_lines):
                    nxt = extracted_lines[k+1]
                    nxt_text = nxt[0]
                    
                    if nxt_text and (nxt_text[0].isupper() or nxt_text[0] in '"“\'') and not re.match(r'^\d+\.', nxt_text):
                        merged_txt = f"{curr_text} {nxt_text}"
                        merged_words = curr[2] + nxt[2]
                        spacer = [{"text": " ", "is_bold": False}]
                        merged_rich = curr[3] + spacer + nxt[3]
                        
                        merged_lines.append((merged_txt, curr[1], merged_words, merged_rich))
                        k += 2
                        continue
                
                merged_lines.append(curr)
                k += 1

            i = 0
            while i < len(merged_lines):
                line_data = merged_lines[i]
                line = line_data[0]
                line_words = line_data[2]
                rich_spans = line_data[3]
                
                metrics['kept_words'] += len(line.split())
                # REMOVED: metrics['total_words'] increment here (now done at page level)

                if line.startswith("<Table_"):
                    current_chunk['content_verbatim'] += f"\n{line}\n"
                    i += 1
                    continue

                match_found = False; found_type = None; match_obj = None
                for h_type in heading_priority:
                    m = regex_map[h_type].match(line)
                    if m:
                        # Safety Checks
                        if h_type == 'content_start_heading' and not m.group(1).isupper(): continue
                        if h_type == 'roman_upper_heading' and not m.group(1).isupper(): continue
                        if h_type == 'preamble_heading' and not m.group(1).isupper(): continue
                        if h_type in ['section_general_heading', 'chapter_heading']:
                            if re.search(r'(below|above)\.?\s*$', line, re.IGNORECASE): continue

                        match_found = True
                        found_type = h_type
                        match_obj = m
                        break
                
                if found_type == 'toc_start_heading':
                    if current_chunk['content_verbatim'].strip(): 
                        all_chunks.append(current_chunk)
                        chunk_counter += 1
                    in_toc_mode = True
                    toc_ids_seen = set() 
                    current_chunk = {
                        'chunk_id': chunk_counter, 'clause_id': "TOC", 'parent_id': "ROOT", 'Title': "Table of Contents",
                        'level': 0, 'appendix/annex': "NIL", 
                        'content_verbatim': line + "\n", 'source_page': str(p_num), 'content_words': line_words
                    }
                    i += 1
                    continue 
                
                if in_toc_mode:
                    is_visual_toc_item = re.search(r'(\.{3,}|\s\d+)$', line.strip())
                    should_exit_toc = False
                    
                    if match_found:
                        temp_id = "NIL"
                        if match_obj.groups():
                            if found_type == 'appendix_heading':
                                temp_id = f"{match_obj.group(1)} {match_obj.group(2)}"
                            else:
                                temp_id = match_obj.group(1).strip().strip('.')
                        temp_id_clean = temp_id.lower().replace(" ", "")
                        if temp_id_clean in toc_ids_seen and not is_visual_toc_item: should_exit_toc = True
                        else:
                            toc_ids_seen.add(temp_id_clean)
                            should_exit_toc = False
                    
                    if should_exit_toc:
                        in_toc_mode = False
                        i -= 1 
                    else:
                        current_chunk['content_verbatim'] += line + "\n"
                        current_chunk['content_words'].extend(line_words)
                    i += 1
                    continue

                if match_found:
                    raw_id = "NIL"
                    if match_obj.groups():
                        if found_type == 'appendix_heading':
                            raw_id = f"{match_obj.group(1)} {match_obj.group(2)}"
                        else:
                            raw_id = match_obj.group(1).strip().strip('.')
                    
                    # Strict Filter for Integer Headers
                    if found_type == 'multi_level_heading' and raw_id.isdigit():
                        has_dot = line.strip().startswith(f"{raw_id}.")
                        rest = line.strip()[len(raw_id):].strip()
                        is_clean_title = re.match(r'^[A-Z][A-Z\s\(\)]{2,}$', rest)
                        if not has_dot and not is_clean_title:
                            match_found = False

                    # FMVSS Citation Safety (Bold Check)
                    if found_type == 'fmvss_paragraph':
                        is_id_bold = False
                        if rich_spans:
                            for s in rich_spans[:3]:
                                if raw_id in s['text'] and s['is_bold']:
                                    is_id_bold = True
                                    break
                            if not is_id_bold: match_found = False

                if match_found:
                    is_digit_only = raw_id.isdigit()
                    line_content = re.sub(r'^' + re.escape(raw_id) + r'[\.\s]*', '', line).strip()

                    if is_digit_only and not line_content and len(raw_id) <= 3: 
                        current_chunk['content_verbatim'] += line + " "
                        current_chunk['content_words'].extend(line_words)
                        i += 1
                        continue
                    if is_digit_only and len(raw_id) == 4 and (1900 <= int(raw_id) <= 2050): 
                        current_chunk['content_verbatim'] += line + " "
                        current_chunk['content_words'].extend(line_words)
                        i += 1
                        continue
                    if is_digit_only and len(raw_id) <= 2 and month_pattern.search(line): 
                        current_chunk['content_verbatim'] += line + " "
                        current_chunk['content_words'].extend(line_words)
                        i += 1
                        continue
                    
                    if current_chunk['content_verbatim'].strip(): 
                        all_chunks.append(current_chunk)
                        chunk_counter += 1
                    
                    level = 1
                    if found_type in ['content_start_heading', 'preamble_heading', 'chapter_heading', 'appendix_heading', 'roman_upper_heading']:
                        level = 1
                    elif found_type == 'multi_level_heading':
                        level = raw_id.count('.') + 1
                    elif found_type in ['numeric_paren_heading', 'alpha_paren_heading']:
                        level = parent_stack[-1][0] + 1 if parent_stack else 1
                        if found_type == 'numeric_paren_heading' and raw_id.isdigit() and int(raw_id) > 1:
                            if parent_stack[-1][1].isdigit(): level = parent_stack[-1][0]
                        elif found_type == 'alpha_paren_heading' and len(raw_id) == 1 and raw_id.lower() > 'a':
                             if len(parent_stack[-1][1]) == 1 and parent_stack[-1][1].isalpha(): level = parent_stack[-1][0]

                    if found_type == 'appendix_heading':
                        current_annex_context = raw_id 
                        parent_stack = [(0, "ROOT")] 

                    while len(parent_stack) > 1 and parent_stack[-1][0] >= level:
                        parent_stack.pop()
                    
                    parent_id = parent_stack[-1][1]
                    parent_stack.append((level, raw_id))
                    
                    detected_title = self._extract_title_from_line(rich_spans, raw_id)
                    if detected_title.strip() in [':', '.', '-', '']: detected_title = "NIL"

                    current_chunk = {
                        'chunk_id': chunk_counter, 'clause_id': raw_id, 'parent_id': parent_id,
                        'level': level, 'appendix/annex': current_annex_context, 'Title': detected_title,
                        'content_verbatim': line + "\n", 'source_page': str(p_num), 'content_words': line_words
                    }
                else:
                    if current_chunk is None:
                        current_chunk = {
                            'chunk_id': chunk_counter, 'clause_id': "PRELUDE", 'parent_id': "ROOT", 'level': 0, 'Title': "NIL",
                            'appendix/annex': "NIL", 'content_verbatim': "", 'source_page': str(p_num), 'content_words': [] 
                        }
                    current_chunk['content_verbatim'] += line + " "
                    current_chunk['content_words'].extend(line_words)
                
                i += 1

        if current_chunk and current_chunk['content_verbatim'].strip(): 
            all_chunks.append(current_chunk)
            
        return all_chunks, metrics

    def _parse_paddle_output(self, output, model_type):
        normalized_blocks = []
        if not output: return []
        
        if model_type == "v3":
            paddle_data = output[0].json.get('res', {})
            structure_blocks = paddle_data.get('parsing_res_list', [])
            layout_boxes = paddle_data.get('layout_det_res', {}).get('boxes', [])
            
            for s_block in structure_blocks:
                best_label = s_block.get('block_label', 'text')
                max_iou = 0.0
                s_bbox = s_block.get('block_bbox')
                if not s_bbox: continue
                for l_box in layout_boxes:
                    iou = self._calculate_iou(s_bbox, l_box.get('coordinate'))
                    if iou > max_iou:
                        max_iou = iou
                        best_label = l_box.get('label')
                if max_iou > 0.80:
                    s_block['block_label'] = best_label
                
            for blk in structure_blocks:
                normalized_blocks.append({
                    'label': blk.get('block_label', 'text'),
                    'bbox': blk.get('block_bbox'),
                    'order': blk.get('block_order')
                })

        elif model_type == "vl":
            res_list = list(output)
            if not res_list: return []
            page_res = res_list[0]
            data = page_res.json if hasattr(page_res, 'json') else page_res
            
            raw_blocks = []
            if 'res' in data and 'parsing_res_list' in data['res']:
                raw_blocks = data['res']['parsing_res_list']
            elif 'res' in data and 'layout_det_res' in data['res']:
                raw_blocks = data['res']['layout_det_res']['boxes']
                
            for blk in raw_blocks:
                bbox = blk.get('block_bbox') or blk.get('coordinate') or blk.get('bbox')
                label = blk.get('block_label') or blk.get('label') or 'text'
                if bbox:
                    normalized_blocks.append({
                        'label': label,
                        'bbox': bbox,
                        'order': None 
                    })
        return normalized_blocks    

    def _clean_text(self, text, font_name):
        if not text: return ""
        if 'symbol' in font_name.lower():
            mapping = self._get_symbol_map()
            text = "".join([mapping.get(c, c) for c in text])
        # Fix degrees Celsius formatting often broken in OCR
        text = re.sub(r'(\d+)\s*[\u03b8q0]\s*([CF])', r'\1°\2', text)
        # Fix tolerance symbols
        text = re.sub(r'\s+[\u03c1rp]\s+(?=\d)', r' ± ', text)
        text = unicodedata.normalize('NFKC', text)
        return text

    def _analyze_bundle_with_ai(self, chunk_list, model="gpt-4o-mini"):
        if not self.openai_api_key or not chunk_list: 
            return []

        # Prepare input for AI (ID: Text)
        input_text = ""
        for c in chunk_list:
            input_text += f"ID {c['chunk_id']}: {c['content_verbatim'][:1000]}\n"

        # Prepare Schema for Labels
        label_options = list(self.LABEL_SPECS.keys())
        
        system_prompt = f"""
        You are a regulatory compliance expert. Analyze the provided text chunks.
        
        TASK:
        1. Group consecutive chunks that belong to the same specific requirement or logic.
        2. Assign a 'Text Type' (Regulatory, Supporting, Comment).
        3. Assign a 'Requirement Label' from this list: {json.dumps(label_options)}
           - If none fit perfectly, use "Unclassified".
        4. Write a 'summary' (1 sentence) describing what this specific segment governs (e.g., "Specifies anti-lock braking test procedures").
        5. Extract a 'key_quote': The single most important sentence from the text VERBATIM.

        OUTPUT JSON FORMAT:
        {{
          "segments": [
            {{ 
              "ids": [1, 2], 
              "text_type": "Regulatory", 
              "label": "Test Methods", 
              "summary": "Description of the requirement...",
              "key_quote": "Exact sentence from text."
            }},
            ...
          ]
        }}
        """

        try:
            resp = openai.OpenAI(api_key=self.openai_api_key).chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"CHUNKS:\n{input_text}"}
                ],
                temperature=0.0
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            return data.get("segments", [])
        except Exception as e:
            self.logger.error(f"AI Error: {e}")
            return []

    def _synthesize_summary_for_category(self, category_name, summaries, quotes, model="gpt-4o-mini"):
            if not self.openai_api_key or not summaries:
                return "No summary available.", "No quote available."

            # 1. Synthesize the summary
            summary_input = "\n".join(f"- {s}" for s in summaries)
            summary_prompt = f"Synthesize these individual points about '{category_name}' into one cohesive paragraph that covers the key aspects. Be concise."
            
            try:
                summary_resp = openai.OpenAI(api_key=self.openai_api_key).chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a technical summarizer."},
                        {"role": "user", "content": f"POINTS:\n{summary_input}\n\nSYNTHESIZED SUMMARY:"}
                    ],
                    temperature=0.1,
                    max_tokens=250
                )
                final_summary = summary_resp.choices[0].message.content.strip()
            except Exception:
                final_summary = "Could not generate summary."

            # 2. Select the best quote
            quote_input = "\n".join(f"- \"{q}\"" for q in quotes if q)
            quote_prompt = f"From the following list of quotes for '{category_name}', select the one that best represents the primary, most binding requirement. Return only that single quote, verbatim, without any extra text or quotation marks."

            try:
                quote_resp = openai.OpenAI(api_key=self.openai_api_key).chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a compliance expert who selects the most critical verbatim text."},
                        {"role": "user", "content": f"QUOTES:\n{quote_input}\n\nBEST QUOTE:"}
                    ],
                    temperature=0.0,
                    max_tokens=250
                )
                final_quote = quote_resp.choices[0].message.content.strip()
                # Clean up potential AI artifacts like surrounding quotes
                if final_quote.startswith('"') and final_quote.endswith('"'):
                    final_quote = final_quote[1:-1]

            except Exception:
                final_quote = "Could not select quote."

            return final_summary, final_quote

    def generate_executive_summary(self, analysis_data, model="gpt-4o-mini"):
        if not analysis_data:
            return {}

        # 1. Group Data by Bundle ID
        grouped = defaultdict(list)
        bundle_meta = {} # Store metadata like page numbers for each bundle
        
        # Define Logical Category Order (Used for Title Sorting)
        category_order = [
            "Definitions",
            "Applicability/Scope",
            "Effective Dates/Commencement/Transitional",
            "Type Approval / Approval Evidence / Marking",
            "HMI (Human-Machine Interface)", # New
            "Handbook / User Manual",        # New
            "Documentation",
            "COP (Conformity of Production)",
            "Test Conditions",
            "Test Methods",
            "Test Scenarios",
            "Exemptions & Alternative Procedures",
            "Alternative/Equivalency/References",
            "Enforcement / Penalties",
            "Notes/Preamble/Editorial",
            "Out of Scope",
            "Unclassified"
        ]
        cat_weight = {name: i for i, name in enumerate(category_order)}

        for row in analysis_data:
            bid = row.get("bundle_id", 0)
            grouped[bid].append(row)
            # Capture page string if not already captured
            if bid not in bundle_meta:
                bundle_meta[bid] = row.get("Source Page", "N/A")

        # 2. Sort Bundles Numerically
        sorted_bids = sorted(grouped.keys())
        
        # 3. Construct Final Output Structure
        final_output = {}
        
        for bid in sorted_bids:
            segments = grouped[bid]
            
            # Sort segments inside the bundle by Clause ID (roughly)
            def seg_sort(item):
                c_str = item.get("Clause_Range", "").split(' ')[0]
                parts = []
                for p in c_str.split('.'):
                    if p.isdigit(): parts.append(int(p))
                return parts
            
            segments.sort(key=seg_sort)
            
            # Generate Option 2 Title: "Definitions & Scope"
            # Get unique labels excluding 'Unclassified'
            labels = set(s.get("Requirement Label", "Unclassified") for s in segments)
            if len(labels) > 1 and "Unclassified" in labels:
                labels.remove("Unclassified")
            
            # Sort labels based on Logical Order (cat_weight) instead of Alphabetical
            sorted_labels = sorted(list(labels), key=lambda x: cat_weight.get(x, 999))
            title_str = " & ".join(sorted_labels)
            
            final_output[bid] = {
                "title_suffix": title_str,
                "page_str": bundle_meta.get(bid, "N/A"),
                "segments": segments
            }

        return final_output

    def detect_header_footer_zones(self, doc, sample_pages=20, margin_percent=5):
        # CHANGED: margin_percent default reduced from 15 to 5
        # This prevents dense text near the bottom from being flagged as footer garbage.
        
        page_height = doc[0].rect.height
        top = page_height * (margin_percent / 100.0)
        bottom = page_height * (1 - margin_percent / 100.0)
        header_cands = defaultdict(int)
        footer_cands = defaultdict(int)
        
        limit = min(len(doc), sample_pages)
        for i in range(limit):
            blocks = doc[i].get_text("blocks")
            for b in blocks:
                y_center = (b[1] + b[3]) / 2
                clean = b[4].strip()
                if not clean: continue
                if y_center < top: header_cands[clean] += 1
                elif y_center > bottom: footer_cands[clean] += 1

        thresh = limit * 0.15
        headers = {k for k, v in header_cands.items() if v > thresh}
        footers = {k for k, v in footer_cands.items() if v > thresh}
        return top, bottom, headers, footers

    def _sort_blocks_in_reading_order(self, blocks, page_width):
            """
            Sorts blocks to handle mixed layouts (1-column headers + 3-column text).
            Logic:
            1. Identify 'Anchors' (Wide blocks > 60% page width) like headers/footers.
            2. Use Anchors to split the page into vertical 'Segments'.
            3. Inside a Segment:
            - Group blocks into 'Columns' based on X-coordinates.
            - Sort Columns Left-to-Right.
            - Sort Blocks inside Columns Top-to-Bottom.
            """
            # Filter out tiny/empty blocks to reduce noise
            blocks = [b for b in blocks if b['bbox'][2] - b['bbox'][0] > 1 and b['bbox'][3] - b['bbox'][1] > 1]
            if not blocks: return []

            # Threshold: Blocks wider than 60% of page are considered 'Anchors' (Headers/Footers/Wide Tables)
            wide_thresh = page_width * 0.60
            
            # 1. Initial sort by Y to find vertical sequence
            blocks.sort(key=lambda b: b['bbox'][1])
            
            segments = []
            current_segment_blocks = []
            
            for b in blocks:
                width = b['bbox'][2] - b['bbox'][0]
                is_wide = width > wide_thresh
                
                if is_wide:
                    # If we have accumulated content blocks, push them as a segment first
                    if current_segment_blocks:
                        segments.append({'type': 'content', 'blocks': current_segment_blocks})
                        current_segment_blocks = []
                    # Push the wide anchor as its own segment
                    segments.append({'type': 'anchor', 'blocks': [b]})
                else:
                    current_segment_blocks.append(b)
            
            # Flush remaining content
            if current_segment_blocks:
                segments.append({'type': 'content', 'blocks': current_segment_blocks})
                
            # 2. Sort inside segments
            final_order = []
            for seg in segments:
                if seg['type'] == 'anchor':
                    # Anchors are already sorted by Y
                    final_order.extend(seg['blocks'])
                else:
                    # Content segment: Cluster into Columns
                    # Sort by Left Edge (X0) to identify columns
                    seg_blocks = seg['blocks']
                    seg_blocks.sort(key=lambda b: b['bbox'][0])
                    
                    columns = []
                    if seg_blocks:
                        current_col = [seg_blocks[0]]
                        current_max_x = seg_blocks[0]['bbox'][2]
                        
                        for b in seg_blocks[1:]:
                            # Check if this block starts significantly after the previous column ends
                            # Buffer of 10px handles minor layout drift
                            if b['bbox'][0] > current_max_x - 10: 
                                # New column found
                                columns.append(current_col)
                                current_col = [b]
                                current_max_x = b['bbox'][2]
                            else:
                                # Belongs to current column (handling indentation)
                                current_col.append(b)
                                current_max_x = max(current_max_x, b['bbox'][2])
                        columns.append(current_col)
                    
                    # Sort inside each column by Y (Top-to-Bottom)
                    for col in columns:
                        col.sort(key=lambda b: b['bbox'][1])
                        final_order.extend(col)

            return final_order
    def _is_span_bold(self, span):
        # Check flags (bit 4 is usually bold)
        if span["flags"] & 2 ** 4:
            return True
        
        # Check font name string for keywords
        font_name = span["font"].lower()
        if any(x in font_name for x in ["bold", "black", "heavy", "demi"]):
            return True
            
        return False


    def extract_content_with_stats(self, doc, num_pages_to_keep, paddle_model, model_type="v3"):
        logical_lines = []
        metrics = {
            "total_words": 0, "kept_words": 0, "dropped_words": 0, 
            "debug_images": {}, "detected_tables": defaultdict(list)
        }
        
        limit = min(len(doc), num_pages_to_keep)
        
        for p_idx in range(limit):
            page = doc[p_idx]
            p_num = p_idx + 1
            
            # 1. Visual Prep
            zoom = 2.0 if model_type == "vl" else 1.5 
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            # 2. Paddle Prediction
            layout_results = paddle_model.predict(img_bgr)
            raw_blocks = self._parse_paddle_output(layout_results, model_type)
            
            # 3. Filter & Sort
            # --- EXCLUSION LIST IS HERE ---
            excluded_labels = ['header', 'footer', 'footnote', 'number', 'page_number']
            valid_blocks = [b for b in raw_blocks if b['label'].lower() not in excluded_labels]
            # ------------------------------
            valid_blocks.sort(key=self._get_sorting_key)
            
            # 4. Debug Visualization
            viz_image = img_bgr.copy()
            scale_x = page.rect.width / img_bgr.shape[1]
            scale_y = page.rect.height / img_bgr.shape[0]
            
            for idx, block in enumerate(raw_blocks):
                label = block['label'].lower()
                bbox = block['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw Box
                is_excl = label in ['header', 'footer']
                color = (0, 0, 255) if 'table' in label else ((128, 128, 128) if is_excl else (0, 255, 0))
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_image, f"{idx+1} {label.upper()}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 5. Text Extraction
                if not is_excl:
                    pdf_rect = fitz.Rect(bbox[0]*scale_x, bbox[1]*scale_y, bbox[2]*scale_x, bbox[3]*scale_y)
                    
                    if 'table' in label:
                        # Register table for AI extraction later
                        metrics['detected_tables'][p_num].append(pdf_rect)
                        # Add placeholder to stream
                        logical_lines.append({
                            "text": f"[[TABLE_DETECTED_PAGE_{p_num}]]",
                            "words": [{"page": p_num, "text": "TABLE", "bbox": list(pdf_rect)}],
                            "page": p_num,
                            "rich_spans": []
                        })
                    else:
                        # Extract text with rich properties inside the block
                        block_dict = page.get_text("dict", clip=pdf_rect)
                        for b in block_dict.get("blocks", []):
                            if "lines" not in b: continue
                            for l in b["lines"]:
                                line_text_parts = []
                                line_words = []
                                line_rich = []
                                
                                for s in l["spans"]:
                                    txt = self._clean_text(s["text"], s["font"])
                                    if txt.strip():
                                        line_text_parts.append(txt)
                                        is_bold = self._is_span_bold(s)
                                        line_rich.append({"text": txt, "is_bold": is_bold})
                                        line_words.append({
                                            "page": p_num, "text": txt, "bbox": [round(x) for x in s["bbox"]]
                                        })
                                
                                full_line = " ".join(line_text_parts).strip()
                                if full_line:
                                    logical_lines.append({
                                        "text": full_line,
                                        "words": line_words,
                                        "page": p_num,
                                        "rich_spans": line_rich
                                    })
                                    metrics['kept_words'] += len(full_line.split())

            # Encode Debug Image for UI
            success, buffer = cv2.imencode(".jpg", viz_image)
            if success:
                metrics['debug_images'][p_num] = buffer.tobytes()

        return logical_lines, metrics

    def _extract_title_from_line(self, rich_spans, clause_id):
        if not rich_spans:
            return "NIL"
            
        title_parts = []
        capturing = False
        
        for i, s in enumerate(rich_spans):
            text_clean = s['text'].strip()
            if not text_clean: continue
            
            is_bold = s['is_bold']
            is_quote = text_clean in ['"', "'", '“', '”']
            
            # Simple heuristic: If it's bold, it's likely part of the title
            if is_bold:
                capturing = True
                title_parts.append(text_clean)
            elif is_quote and capturing:
                # Keep end quotes attached to title
                pass
            elif is_quote and not capturing:
                # Skip start quotes if not bold (rare)
                 pass
            else:
                # Stop if we hit non-bold text (often the content starting on same line)
                if capturing:
                    break
                else:
                    return "NIL"
        
        if not title_parts:
            # Fallback: If no bold text found, check if the whole line is the title
            # This happens if the PDF doesn't use bold fonts for headers
            full_text = " ".join([s['text'] for s in rich_spans]).strip()
            if len(full_text) < 100: # Arbitrary length limit for a title
                raw_header = full_text
            else:
                return "NIL"
        else:
            raw_header = " ".join(title_parts).strip()
        
        # Remove the ID from the Title string if present
        if clause_id and clause_id != "NIL":
            # Escape the ID for regex (e.g. "5.1")
            pattern = r'^' + re.escape(clause_id) + r'[\.\s]*'
            clean_title = re.sub(pattern, '', raw_header).strip()
        else:
            clean_title = raw_header

        clean_title = clean_title.strip('"\'“”')
        clean_title = clean_title.rstrip('.')
        
        if not clean_title:
            return "NIL"
            
        return clean_title

    def get_compliance_patterns(self):
        return {
            'toc_start_heading': r'(?i)^\s*(?:\d+\.?\s+)?(TABLE\s+OF\s+CONTENTS|CONTENTS)\s*$',
            'preamble_heading': r'^\s*((?:AGENCY|ACTION|SUMMARY|DATES|ADDRESSES|SUPPLEMENTARY INFORMATION))\s*:?\s*(.*)',
            'chapter_heading': r'^\s*(?i:CHAPTER)\s+([A-Z0-9\-\.\s]+).*',
            'section_general_heading': r'^\s*(?i:SECTION)\s+([A-Z0-9\-\.\s]+).*',
            'cfr_part_heading': r'^\s*((?:49\s+CFR\s+)?PART\s+\d+).*',
            'cfr_section_heading': r'^\s*(§\s*\d+(\.\d+)?)\s+(.*)',
            'fmvss_paragraph': r'^\s*(S\d+(\.\d+)*)\s+(.*)',
            'article_heading': r'^\s*((?i:ARTICLE\s+\d+))\b\.?\s*(?:[A-Z0-9"“\'].*)?$',
            'appendix_heading': r'^\s*(APPENDIX|ANNEX)\s+([A-Z0-9]+)\s*(.*)',
            'roman_upper_heading': r'^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\.\s+(?![A-Z0-9]\.)(.*)',
            'multi_level_heading': r'^\s*(?!49\s+CFR)["“\']?(\d{1,3}(?!\d)(\.\d+)*)\.?\s*(?:[A-Z0-9"“\'].*)?$',
            'numeric_paren_heading': r'^\s*\((\d+)\)\s*(?![\d]{3}[-\s–\u2013\u2014])(.*)',
            'alpha_paren_heading': r'^\s*\(([a-z])\)\s+(.*)',
        }

    def _extract_candidates_from_text(self, text):
        candidates = []
        
        # 1. Standard Documents (Annex, Appendix, etc.)
        label_pattern = re.compile(
            r'(?i)\b(Annex(?:es)?|Append(?:ix|ices)|Tab(?:le|les)|Fig(?:ure|ures)|Sec(?:tion|tions)?|Art(?:icle|icles)?|Part|Clause|Diag(?:ram)?|Sheet)\s+'
            r'((?:(?!(?:Annex|Append|Tab|Fig|Sec|Art|Part|Clause|Diag|Sheet))[A-Z0-9\.\-]+(?:-[A-Z0-9\.\-]+)?(?:,\s*|,\s*and\s*|\s+and\s+)?)+)',
            re.IGNORECASE
        )
        for match in label_pattern.finditer(text):
            raw_type = match.group(1).lower()
            content_blob = match.group(2)
            
            # Normalize prefix
            prefix = raw_type.capitalize()
            if raw_type.startswith('annex'): prefix = "Annex"
            elif raw_type.startswith('append'): prefix = "Appendix"
            elif raw_type.startswith('tab'): prefix = "Table"
            elif raw_type.startswith('fig'): prefix = "Figure"
            elif raw_type.startswith('sec'): prefix = "Section"
            elif raw_type.startswith('art'): prefix = "Article"
            elif raw_type.startswith('part'): prefix = "Part"
            elif raw_type.startswith('clause'): prefix = "Clause"
            elif raw_type.startswith('diag'): prefix = "Diagram"
            elif raw_type.startswith('sheet'): prefix = "Sheet"
            
            items = re.split(r',?\s*(?:and\s+)?', content_blob)
            for item in items:
                clean_item = item.strip().strip('.')
                if not clean_item: continue
                if len(clean_item) == 1 and clean_item.islower(): continue
                candidates.append(f"{prefix} {clean_item}")

        # 2. Strict Numeric References (e.g. "3.1", "5.2.1")
        dotted_pattern = re.compile(r'\b(\d+(?:\.\d+)+)\b')
        ignore = {'mg', 'kg', 'mm', 'cm', 'm', 'g', 'hz', 'v', 'cfr', 'usc', 'iso', 'astm'}
        for match in dotted_pattern.finditer(text):
            val = match.group(1)
            next_chunk = text[match.end():match.end()+10].strip().lower()
            first_word = next_chunk.split()[0].strip('.,;:)') if next_chunk else ""
            if first_word in ignore: continue
            candidates.append(val)

        # 3. Generic Paragraph References (e.g. "paragraph 2.3")
        generic_pattern = re.compile(r'(?i)\b(?:paragraph|item|point|para)\s+([A-Z0-9\.]+)\b')
        for match in generic_pattern.finditer(text):
            candidates.append(match.group(1).strip('.'))
            
        # --- NEW: US FMVSS References (e.g. "S5.1", "S14.2") ---
        # Look for Capital S followed immediately by a number
        fmvss_pattern = re.compile(r'\b(S\d+(?:\.\d+)*)\b')
        for match in fmvss_pattern.finditer(text):
            candidates.append(match.group(1).strip('.'))

        # --- NEW: CFR Section References (e.g. "§ 571.111") ---
        cfr_pattern = re.compile(r'(?:§|49\s+CFR)\s*(\d+(?:\.\d+)*)')
        for match in cfr_pattern.finditer(text):
            # Normalizing to just the number often matches better with ID logic
            # Or keep the symbol if your IDs use it. 
            # Given our parsing logic extracted "§ 571.111", we should capture full string.
            full_ref = f"§ {match.group(1)}"
            candidates.append(full_ref)
            
        return list(set(candidates))

    def _resolve_references(self, chunks):
        valid_ids = {c.get('clause_id', '').strip().lower() for c in chunks if c.get('clause_id') != "NIL"}
        for chunk in chunks:
            text = chunk.get('content_verbatim', '')
            cid = chunk.get('clause_id', '').strip().lower()
            raw = self._extract_candidates_from_text(text)
            confirmed = []
            if chunk.get('clause_id') == "TOC":
                for c in raw:
                    # In TOC, we primarily care about high-level structure
                    if c.lower().startswith(('annex', 'appendix', 'part')): confirmed.append(c)
            else:
                for c in raw:
                    cl = c.strip().lower()
                    # EXTENDED WHITELIST: Added diagram, sheet, part, appendix, annex to valid visual types
                    is_vis = cl.startswith(('table', 'figure', 'fig', 'chart', 'diagram', 'sheet', 'part', 'appendix', 'annex'))
                    
                    # Check if it exists in the document structure OR is a known visual type
                    is_valid = cl in valid_ids
                    
                    # Add if valid and not a self-reference
                    if (is_vis or is_valid) and (not cid or cl != cid):
                        confirmed.append(c)
            chunk['references'] = sorted(list(set(confirmed)))
        return chunks

    def reconstruct_split_headers(self, lines):
        if not lines: return []
        merged = []
        i = 0
        while i < len(lines):
            curr = lines[i]
            txt = curr['text'].strip()
            
            # Check for short numeric or roman text
            is_num = len(txt) < 10 and re.match(r'^[\(\d][\d\.\)\w]+$', txt) and not re.search(r'[a-zA-Z]{3,}', txt)
            is_rom = re.match(r'^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|[A-Z])\.$', txt)
            
            if (is_num or is_rom) and (i + 1 < len(lines)):
                nxt = lines[i+1]
                nxt_txt = nxt['text'].strip()
                
                # Check if next line looks like a continuation
                is_list_item = re.match(r'^\s*\(?[\da-zA-Z]+\)', nxt_txt)
                
                if nxt_txt and (nxt_txt[0].isupper() or nxt_txt[0] in '"“\'' or nxt_txt[0].isdigit()) and not is_list_item:
                    # MERGE LOGIC UPDATED to include rich_spans
                    merged.append({
                        "text": f"{txt} {nxt_txt}", 
                        "words": curr['words'] + nxt['words'], 
                        "page": curr['page'],
                        "rich_spans": curr.get('rich_spans', []) + nxt.get('rich_spans', []) # <--- ADDED
                    })
                    i += 2
                    continue
            
            merged.append(curr)
            i += 1
        return merged

    def parse_structure(self, lines, patterns):
        chunks = []
        current = None
        seen = set()
        mode = 'default'
        toc_titles = set()
        toc_ids = set() 
        parent_stack = [(-2, "ROOT")] 
        
        # --- NEW FLAG: Track if we have entered the main document structure ---
        # This prevents Bold text on the cover page (PRELUDE) from splitting chunks.
        structure_started = False 
        
        ranks = {
            'cfr_part_heading': -1, 'section_heading': -1, 'appendix_heading': -1, 'article_heading': -1,
            'preamble_heading': 0, 'cfr_section_heading': 0, 'supplementary_info': 0, 'roman_upper_heading': 0,
            'multi_level_heading': 0, 'multi_level_content_heading': 0,
            'fmvss_paragraph': 1, 'alpha_upper_heading': 1, 'definition_heading': 1,
            'implicit_heading': 2, 
            'numeric_paren_heading': 2, 'alpha_paren_heading': 3, 'simple_list_item': 4,
            'roman_lower_paren_heading': 4, 'alpha_closing_paren_heading': 5,
        }
        
        toc_start = re.compile(patterns.get('toc_start_heading', r'^\s*CONTENTS\s*$'), re.IGNORECASE)
        
        patterns['multi_level_heading'] = r'^\s*(\d{1,3}(?:\.\d+)*)\.?\s+([A-Z][A-Z\s]{2,})$'
        patterns['multi_level_content_heading'] = r'^\s*(\d{1,3}(?:\.\d+)*\.)\s+(.+)$'
        
        heading_order = [
            'preamble_heading', 'cfr_part_heading', 'cfr_section_heading', 'fmvss_paragraph', 
            'implicit_heading', 'section_heading', 'multi_level_heading', 'multi_level_content_heading',          
            'appendix_heading', 'article_heading', 'supplementary_info', 
            'roman_upper_heading', 'alpha_upper_heading', 'definition_heading', 
            'numeric_paren_heading', 'alpha_paren_heading', 'roman_lower_paren_heading'
        ]
        heading_res = {k: re.compile(patterns[k]) for k in heading_order if k in patterns}

        def finalize(c):
            if c and c.get('content_verbatim', '').strip(): chunks.append(c)

        def norm_title(t):
            t = re.split(r'\s*\.{3,}', t)[0].strip()
            t = re.sub(r'[\s\d\(\)¹²³⁴⁵⁶⁷⁸⁹]+$', '', t).strip()
            m = re.search(r'[a-zA-Z]', t)
            return t[m.start():].lower().strip() if m else t.lower().strip()

        def is_visual_header(rich_spans):
            if not rich_spans: return False
            bold_count = sum(len(s['text']) for s in rich_spans if s['is_bold'])
            total_count = sum(len(s['text']) for s in rich_spans)
            if total_count == 0: return False
            return (bold_count / total_count) > 0.8

        i = 0
        while i < len(lines):
            line_obj = lines[i]
            text = line_obj['text']

            if mode == 'default':
                if toc_start.match(text):
                    structure_started = True # <--- TOC starts structure
                    finalize(current)
                    mode = 'in_toc'
                    toc_titles = set()
                    toc_ids = set()
                    current = {
                        "clause_id": "TOC", "Title": "Table of Contents", "parent_id": "ROOT", 
                        "content_verbatim": text+"\n", "content_words": list(line_obj['words'])
                    }
                
                else:
                    match, h_type = None, None
                    for k in heading_order:
                        if k not in heading_res: continue
                        m = heading_res[k].match(text)
                        if m:
                            match, h_type = m, k
                            break
                    
                    if match:
                        structure_started = True # <--- Numbered Header starts structure
                        finalize(current)
                        
                        raw_cid = match.group(1).strip().strip('.')
                        if h_type == 'implicit_heading':
                            slug = "_".join(raw_cid.split()[:4])
                            raw_cid = slug
                        
                        rank = ranks.get(h_type, 99)
                        if h_type in ['multi_level_heading', 'multi_level_content_heading']:
                            rank = raw_cid.count('.')
                        elif h_type == 'fmvss_paragraph':
                            rank = raw_cid.count('.') + 1
                        
                        while parent_stack and parent_stack[-1][0] >= rank: parent_stack.pop()
                        
                        pid = parent_stack[-1][1] if parent_stack else "ROOT"
                        cid = raw_cid

                        parent_stack.append((rank, cid))
                        seen.add(cid) 

                        detected_title = self._extract_title_from_line(line_obj.get('rich_spans', []), raw_cid)

                        current = {
                            "clause_id": cid, 
                            "Title": detected_title, 
                            "parent_id": pid, 
                            "level": len(parent_stack)-1, 
                            "content_verbatim": text+"\n", 
                            "content_words": list(line_obj['words'])
                        }
                    
                    # --- FIX IS HERE: Only check Visual Headers if structure has started ---
                    elif structure_started and is_visual_header(line_obj.get('rich_spans', [])):
                        finalize(current)
                        current = {
                            "clause_id": "NIL",
                            "Title": text, 
                            "parent_id": parent_stack[-1][1] if parent_stack else "ROOT",
                            "level": len(parent_stack),
                            "content_verbatim": text + "\n",
                            "content_words": list(line_obj['words'])
                        }
                    # ---------------------------------------------------------------------
                    
                    elif text.strip():
                        if current is None: 
                            # This ensures the first chunk is named PRELUDE and collects all cover page text
                            current = {
                                "clause_id": "PRELUDE", "Title": "NIL", "parent_id": "ROOT", 
                                "level": 0, "content_verbatim": "", "content_words": []
                            }
                        current['content_verbatim'] += text + "\n"
                        current['content_words'].extend(line_obj['words'])

            elif mode == 'in_toc':
                current['content_verbatim'] += text + "\n"
                current['content_words'].extend(line_obj['words'])
                
                is_heading = False
                found_id = None
                for k in heading_order:
                    m = heading_res[k].match(text)
                    if m: 
                        is_heading = True; found_id = m.group(1).strip().strip('.')
                        break
                if is_heading and found_id and found_id not in toc_ids:
                    toc_ids.add(found_id)
                elif is_heading and found_id in toc_ids:
                    finalize(current)
                    mode = 'default'
                    current = None
                    parent_stack = [(-2, "ROOT")]
                    i -= 1

            i += 1
        finalize(current)
        return chunks

    def assign_bundle_ids(self, chunks):
        bid = 1
        cid_map = {c['clause_id'].lower(): c for c in chunks if c.get('clause_id')}
        
        while True:
            unassigned = [c for c in chunks if 'bundle_id' not in c]
            if not unassigned: break
            
            groups = defaultdict(list)
            for c in unassigned: groups[c['parent_id']].append(c)
            
            largest = []
            non_root = {k:v for k,v in groups.items() if k!="ROOT"}
            if non_root: largest = max(non_root.values(), key=len)
            
            if len(largest) < 2: break
            
            current_bundle = list(largest)
            cids = {c['clause_id'] for c in current_bundle}
            q = list(largest)
            
            while q:
                curr = q.pop(0)
                for ref in curr.get('references', []):
                    target = cid_map.get(ref.lower())
                    if target and 'bundle_id' not in target and target['clause_id'] not in cids:
                        current_bundle.append(target)
                        cids.add(target['clause_id'])
                        q.append(target)
            
            for c in current_bundle: c['bundle_id'] = bid
            bid += 1
            
        for c in [c for c in chunks if 'bundle_id' not in c]:
            c['bundle_id'] = bid; bid += 1
        return chunks

    def _merge_rects(self, rects, thresh=30):
        if not rects: return []
        rects.sort(key=lambda r: r.y0)
        merged = []
        while rects:
            curr = rects.pop(0)
            i = 0
            while i < len(rects):
                other = rects[i]
                expanded = fitz.Rect(curr.x0-thresh, curr.y0-thresh, curr.x1+thresh, curr.y1+thresh)
                if expanded.intersects(other):
                    curr |= other
                    rects.pop(i)
                else: i += 1
            merged.append(curr)
        return merged

    def extract_complex_elements(self, doc, chunks):
        page_map = defaultdict(list)
        for c in chunks:
            c['tables_html'], c['images'] = [], []
            if c.get('content_words'):
                y_c = sum(w['bbox'][1] for w in c['content_words'])/len(c['content_words'])
                page_map[c['content_words'][0]['page']-1].append({'chunk': c, 'y': y_c})

        for p_idx in range(len(doc)):
            chunk_data = page_map.get(p_idx, [])
            if not chunk_data: continue
            page = doc[p_idx]
            
            excl_rects = []
            try:
                for t in page.find_tables(strategy="lines"):
                    if t.row_count > 1 and t.col_count > 1:
                        ty = (t.bbox[1]+t.bbox[3])/2
                        closest = min(chunk_data, key=lambda x: abs(x['y']-ty))
                        # --- FIX IS HERE ---
                        # Replaced deprecated 'applymap' with 'map'
                        df = t.to_pandas().map(lambda x: x.replace('\n', '<br>') if isinstance(x,str) else x)
                        closest['chunk']['tables_html'].append(df.to_html(index=False, header=True, border=1, escape=False).replace('\n', ''))
                        excl_rects.append(fitz.Rect(t.bbox))
            except: pass

            cands = []
            for img in page.get_images(full=True):
                try: cands.append({'type': 'raster', 'rect': page.get_image_bbox(img), 'xref': img[0]})
                except: continue
            
            raw_draw = [p['rect'] for p in page.get_drawings() if p['fill'] != (1.,1.,1.) and p['color'] is not None and p['rect'].width>5]
            for d in self._merge_rects(raw_draw):
                if d.width > 20 and d.height > 20 and len(page.get_text("text", clip=d).strip()) < 500:
                    cands.append({'type': 'vector', 'rect': d})

            for cand in cands:
                r = cand['rect']
                is_excl = False
                for ex in excl_rects:
                    if (r & ex).get_area() > 0.5 * r.get_area(): is_excl = True; break
                if is_excl: continue

                winner = min(chunk_data, key=lambda x: abs(x['y'] - (r.y0+r.y1)/2))
                try:
                    b64 = None
                    mime = "image/png"
                    if cand['type'] == 'raster':
                        base = doc.extract_image(cand['xref'])
                        b64 = base64.b64encode(base["image"]).decode('utf-8')
                        mime = f"image/{base['ext']}"
                    else:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2,2), clip=r)
                        if pix.width > 0: b64 = base64.b64encode(pix.tobytes("png")).decode('utf-8')
                    
                    if b64 and not any(i['data']==b64 for i in winner['chunk']['images']):
                        winner['chunk']['images'].append({"page": p_idx+1, "type": cand['type'], "mime_type": mime, "data": b64})
                except: continue
        return chunks

    def enrich_and_finalize(self, chunks, ext_table_pages=None):
        final = []
        curr_cont = "NIL"
        
        for i, c in enumerate(chunks):
            full = c.get('content_verbatim', '').strip()
            cid = c.get('clause_id', 'NIL')
            upper_cid = cid.upper()
            
            if "ANNEX" in upper_cid or "APPENDIX" in upper_cid:
                curr_cont = cid
                
            if cid in ['TOC', 'PRELUDE', 'NIL']:
                lvl = 0
            else:
                lvl = cid.count('.') + 1

            raw_pages = [w['page'] for w in c.get('content_words', [])]
            if raw_pages:
                unique_pages = sorted(list(set(raw_pages)))
                page_str = ", ".join(map(str, unique_pages))
            else:
                page_str = ""

            final.append({
                "chunk_id": i + 1,
                "clause_id": cid, 
                "Title": c.get('Title', 'NIL'),
                "parent_id": c.get('parent_id', 'ROOT'),
                "Annex/Appendix": curr_cont,
                "level": lvl,
                "references": c.get('references', []),
                "bundle_id": c.get('bundle_id', -1),
                "Source Page": page_str,
                "content_verbatim": full,
                "tables_html": c.get('tables_html', []),
                "images": c.get('images', [])
            })
            
        return final

    def _classify_bundle(self, bundle_text):
        if not self.openai_api_key or not bundle_text.strip(): return "Unclassified"
        
        # USE KEYS FROM BUNDLE_CONFIG
        categories_list = "\n".join([f"- {k}" for k in self.BUNDLE_CONFIG.keys()])
        
        sys_p = "You are a text classification expert. Categorize the regulation text into one of the provided categories. Return ONLY the category name."
        user_p = f"CATEGORIES:\n{categories_list}\n\nTEXT:\n{bundle_text[:4000]}\n\nBest Category:"
        
        try:
            resp = openai.OpenAI(api_key=self.openai_api_key).chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                temperature=0.0
            )
            return resp.choices[0].message.content.strip()
        except: return "Unclassified"

    def _summarize_bundle(self, bundle_text, label):
        if not self.openai_api_key or not bundle_text.strip(): return {"error": "No text"}
        
        # USE INSTRUCTIONS FROM BUNDLE_CONFIG
        instructions = self.BUNDLE_CONFIG.get(label, self.BUNDLE_CONFIG["Unclassified"])
        
        prompt_fields = []
        for key, text_instruction in instructions.items():
            prompt_fields.append(f'"{key}": "{text_instruction}"')
        
        json_structure = "{\n  " + ",\n  ".join(prompt_fields) + "\n}"
        
        sys_p = "You are a legal summarizer. Extract information based on the instructions. Return valid JSON."
        user_p = f"TEXT:\n{bundle_text[:8000]}\n\nOUTPUT JSON FORMAT:\n{json_structure}"
        
        try:
            resp = openai.OpenAI(api_key=self.openai_api_key).chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                temperature=0.0
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e: return {"error": str(e)}

    def generate_bundle_summaries(self, chunks):
        if not self.openai_api_key: return []
        
        bundles = defaultdict(list)
        for c in chunks:
            if 'bundle_id' in c: bundles[c['bundle_id']].append(c['content_verbatim'])
        
        summaries = []
        st.write("🤖 Generating Bundle Summaries...")
        bar = st.progress(0, text="Summarizing...")
        total = len(bundles)
        
        for i, (bid, content) in enumerate(sorted(bundles.items())):
            bar.progress((i+1)/total, text=f"Processing Bundle {bid}/{total}")
            text = "\n\n".join(content)
            label = self._classify_bundle(text)
            
            # Map LLM label back to config keys if slightly off
            if label not in self.BUNDLE_CONFIG:
                # simple fuzzy match fallback
                for key in self.BUNDLE_CONFIG:
                    if label in key or key in label:
                        label = key
                        break
            
            summary_data = self._summarize_bundle(text, label)
            summaries.append({"bundle_id": bid, "classified_label": label, "summary": summary_data})
            
        bar.empty()
        return summaries

    def generate_bundle_analysis(self, chunks, model="gpt-4o-mini"):
        if not self.openai_api_key: return []
        
        chunk_map = {c['chunk_id']: c for c in chunks}
        for c in chunks:
            c['Requirement Label'] = "Unclassified"
            c['Text Type'] = "Unclassified"

        bundles = defaultdict(list)
        for c in chunks:
            if 'bundle_id' in c: bundles[c['bundle_id']].append(c)
        
        summary_data_for_rollup = []
        
        st.write("🤖 Analyzing Bundles...")
        bar = st.progress(0, text="Analyzing...")
        total = len(bundles)
        
        for i, (bid, b_chunks) in enumerate(sorted(bundles.items())):
            bar.progress((i+1)/total, text=f"Processing Bundle {bid}/{total}")
            
            b_chunks.sort(key=lambda x: x['chunk_id'])
            
            # extract pages for this bundle
            all_pages = set()
            for c in b_chunks:
                if c.get('Source Page'):
                    all_pages.update(p.strip() for p in c['Source Page'].split(',') if p.strip())
            # Sort pages numerically
            sorted_pages = sorted(all_pages, key=lambda x: int(x) if x.isdigit() else 9999)
            page_str = ", ".join(sorted_pages) if sorted_pages else "N/A"

            ai_segments = self._analyze_bundle_with_ai(b_chunks, model)
            
            if ai_segments:
                for seg in ai_segments:
                    target_ids = seg.get("ids", [])
                    label = seg.get("label", "Unclassified")
                    t_type = seg.get("text_type", "Unclassified")
                    summary = seg.get("summary", "")
                    quote = seg.get("key_quote", "")
                    
                    valid_ids_for_summary = []
                    
                    for tid in target_ids:
                        if tid in chunk_map:
                            if chunk_map[tid]['Requirement Label'] == "Unclassified":
                                chunk_map[tid]['Requirement Label'] = label
                                chunk_map[tid]['Text Type'] = t_type
                                valid_ids_for_summary.append(tid)
                    
                    # Capture the segment for the report even if valid_ids_for_summary is empty
                    # (Sometimes AI returns a summary for a chunk that was already labeled by a previous pass)
                    # To be safe, we calculate range from the raw target_ids if they exist in the bundle
                    ids_in_bundle = [tid for tid in target_ids if tid in chunk_map and chunk_map[tid]['bundle_id'] == bid]
                    
                    if ids_in_bundle:
                        ids_in_bundle.sort()
                        first_id = ids_in_bundle[0]
                        last_id = ids_in_bundle[-1]
                        c_start = chunk_map[first_id].get('clause_id', 'NIL')
                        c_end = chunk_map[last_id].get('clause_id', 'NIL')
                        c_range = c_start if c_start == c_end else f"{c_start} - {c_end}"

                        summary_data_for_rollup.append({
                            "bundle_id": bid,           # <--- NEW FIELD
                            "Source Page": page_str,    # <--- NEW FIELD
                            "Requirement Label": label,
                            "Summary_Text": summary,
                            "Key_Quote": quote,
                            "Clause_Range": c_range
                        })

        bar.empty()
        return summary_data_for_rollup

    def run_pipeline(self, pdf_bytes, num_pages, enable_ai, ai_thresh, table_model, analysis_model, api_key, do_summary, paddle_model, model_type):
        self.openai_api_key = api_key
        try: doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e: st.error(f"PDF Error: {e}"); return [], None, None, None

        with st.status("🚀 Processing...", expanded=True) as status:
            
            # --- VISION PROGRESS BAR ---
            st.write("👁️ Running Vision Layout Analysis & Structure Logic...")
            vision_bar = st.progress(0, text="Initializing Vision Scan...")
            
            def update_vision_bar(progress):
                vision_bar.progress(progress, text=f"Scanning Page {int(progress * num_pages)}/{num_pages}")
            
            chunks, metrics = self.hybrid_extract_and_structure(doc, num_pages, paddle_model, model_type, progress_callback=update_vision_bar)
            vision_bar.empty() # Remove bar when done
            # ---------------------------

            chunks = self.extract_complex_elements(doc, chunks)
            chunks = self._resolve_references(chunks)
            chunks = self.assign_bundle_ids(chunks)
            
            ext_pages = set()
            
            # 4. AI Table Extraction (Targeted via Paddle Detection)
            if enable_ai and api_key and metrics['detected_tables']:
                total_tables = sum(len(v) for v in metrics['detected_tables'].values())
                st.write(f"📊 AI Extracting {total_tables} detected tables...")
                
                # --- TABLE PROGRESS BAR ---
                table_bar = st.progress(0, text="Queueing tables...")
                # --------------------------

                te = TableExtractor(api_key, model=table_model)
                tasks = []
                for p_num, rects in metrics['detected_tables'].items():
                    for r in rects:
                        tasks.append((doc[p_num-1], p_num, r))
                
                results = {}
                completed_tables = 0
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_task = {executor.submit(te._extract_page_sync, t): t[1] for t in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        # --- UPDATE TABLE BAR ---
                        completed_tables += 1
                        table_bar.progress(completed_tables / total_tables, text=f"Extracting Table {completed_tables}/{total_tables}")
                        # ------------------------
                        try:
                            p_num, tables = future.result()
                            if tables: 
                                if p_num not in results: results[p_num] = []
                                results[p_num].extend(tables)
                        except Exception: pass
                
                table_bar.empty() # Remove bar when done
                ext_pages = set(results.keys())
                st.session_state['ai_table_report_data'] = results

            final = self.enrich_and_finalize(chunks, ext_pages)
            
            for c in final:
                if 'Requirement Label' not in c: c['Requirement Label'] = "Unclassified"
                if 'Text Type' not in c: c['Text Type'] = "Unclassified"
            
            summary_data = [] 
            if do_summary and api_key:
                raw_summary_inputs = self.generate_bundle_analysis(final, model=analysis_model)
                if raw_summary_inputs:
                    summary_data = self.generate_executive_summary(raw_summary_inputs, model=analysis_model)
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
        
        return final, metrics, None, summary_data

st.set_page_config(layout="wide", page_title="AI PDF Structure Extractor")
st.title("📄 PaddleOCR Powered PDF Structure Extractor")
st.markdown("Upload a PDF to extract content, cluster bundles, and generate AI-classified requirements.")

# [Sidebar Configuration Update]
with st.sidebar:
    st.header("⚙️ Configuration")
    f_up = st.file_uploader("1. Choose PDF", type="pdf")
    max_p = 10
    if f_up:
        with fitz.open(stream=f_up.getvalue(), filetype="pdf") as d: max_p = d.page_count
        f_up.seek(0)
    
    n_pages = st.number_input("Pages to process", 1, max_p, max_p)
    
    # --- NEW: Model Selector ---
    ocr_model_type = st.selectbox("Vision Model", ["vl", "v3"], index=1, help="'vl' is slower but better at complex layouts.")
    
    st.markdown("---")
    st.header("🤖 AI Features")
    ai_tables = st.checkbox("AI Table Extraction (Image)", False)
    ai_sum = st.checkbox("Bundle Analysis (Text)", False)
    
    analysis_model = "gpt-4o-mini"
    if ai_sum:
        analysis_model = st.selectbox("Analysis Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    
    api_key = ""
    if ai_tables or ai_sum:
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key: api_key = os.environ.get("OPENAI_API_KEY", "")
    
    st.markdown("---")
    btn = st.button("Process Document", type="primary")

# [Main Logic Update]
if btn and f_up:
    # --- CRITICAL MEMORY CLEANUP ---
    # Clear Streamlit State
    for k in ['ai_table_report_data', 'results', 'df_final', 'metrics', 'bundle_analysis', 'executive_summary', 'pdf_bytes']:
        if k in st.session_state: del st.session_state[k]

    # Force Python Garbage Collection
    gc.collect()

    # Force Paddle/CUDA Memory Cleanup
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except: pass

    # Load selected model type
    with st.spinner(f"Initializing {ocr_model_type.upper()} Vision Engine..."):
        paddle = load_paddle_model(ocr_model_type)

    if not paddle:
        st.error("Failed to load Vision Model. Check installation.")
    else:
        processor = DocumentProcessor()
        st.session_state['pdf_bytes'] = f_up.getvalue()
        
        if (ai_tables or ai_sum) and not api_key:
            st.error("AI features require an API Key.")
        else:
            res, met, _, summary = processor.run_pipeline(
                f_up.getvalue(), n_pages, ai_tables, 110, 
                table_model="gpt-4o", 
                analysis_model=analysis_model, 
                api_key=api_key, 
                do_summary=ai_sum,
                paddle_model=paddle,
                model_type=ocr_model_type # Pass user selection
            )
            
            st.session_state['results'] = res
            st.session_state['metrics'] = met
            st.session_state['bundle_analysis'] = None
            st.session_state['executive_summary'] = summary
            st.session_state['fname'] = f_up.name

            if res:
                df = pd.DataFrame(res)
                drop = ['tables_html', 'content_words', 'images']
                df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
                
                base_order = ['chunk_id', 'clause_id', 'parent_id', 'level', 'appendix/annex', 'Title', 'content_verbatim', 
                              'references', 'bundle_id', 'Source Page']
                
                existing_base = [c for c in base_order if c in df.columns]
                others = [c for c in df.columns if c not in base_order and c not in ['Requirement Label', 'Text Type']]
                final_cols = existing_base + others + ['Requirement Label', 'Text Type']
                final_cols = [c for c in final_cols if c in df.columns]
                
                st.session_state['df_final'] = df[final_cols]

if 'df_final' in st.session_state:
    df = st.session_state['df_final']
    met = st.session_state.get('metrics', {})
    summary_data = st.session_state.get('executive_summary', {})
    
    # --- METRICS CALCULATION START ---
    total_w = met.get('total_words', 0)
    kept_w = met.get('kept_words', 0)
    
    # 1. Structure Yield
    if total_w > 0:
        yield_pct = (kept_w / total_w) * 100
        status_msg = f"Processed {kept_w:,} valid words after filtering ({yield_pct:.1f}% retention)."
    else:
        status_msg = f"Processed {kept_w:,} valid words."

    # 2. AI Coverage (Only if AI ran)
    if 'Requirement Label' in df.columns and kept_w > 0:
        classified_mask = df['Requirement Label'] != "Unclassified"
        classified_words = df.loc[classified_mask, 'content_verbatim'].astype(str).apply(lambda x: len(x.split())).sum()
        ai_pct = (classified_words / kept_w) * 100
        status_msg += f" Of this text, {ai_pct:.1f}% was successfully categorized by the AI model."
    
    st.success(status_msg)
    # --- METRICS CALCULATION END ---
    
    t_names = ["📄 Structure", "📋 Executive Summary", "🔍 Debug", "📊 AI Tables", "📥 Downloads"]
    if not summary_data: t_names.remove("📋 Executive Summary")
    
    tabs = st.tabs(t_names)
    idx = 0
    
    # 1. Structure Tab
    with tabs[idx]:
        st.dataframe(df, width="stretch")
        idx += 1
    
    # 2. Executive Summary Tab (Bundle View)
    if summary_data:
        with tabs[idx]:
            # summary_data format: { bid: {'title_suffix': '...', 'page_str': '...', 'segments': [...]} }
            total_bundles = len(summary_data)
            total_segments = sum(len(v['segments']) for v in summary_data.values())
            
            st.info(f"Generated report covering {total_bundles} Bundles ({total_segments} specific segments).")
            
            # Iterate strictly by Bundle ID
            for bid in sorted(summary_data.keys()):
                data = summary_data[bid]
                title = data['title_suffix']
                pg = data['page_str']
                
                # Bundle Header
                st.markdown(f"### Bundle {bid}: {title} (Page {pg})")
                
                # Render Segments
                for item in data['segments']:
                    lbl = item.get("Requirement Label", "Unclassified")
                    c_range = item.get("Clause_Range", "Ref")
                    summary_text = item.get("Summary_Text", "No summary.")
                    quote = item.get("Key_Quote", "")
                    
                    # Sub-header for the segment (Requirement Label)
                    st.markdown(f"**[{lbl}] {c_range}**")
                    st.markdown(f"{summary_text}")
                    
                    if quote:
                        st.markdown(f"> *Verbatim: “{quote}”*")
                    
                    # Small spacer between segments in same bundle
                    st.caption("") 
                
                st.markdown("---") # Strong Separator between bundles
            
            # Legend
            st.markdown("### 📖 Guide to Requirement Labels")
            temp_proc = DocumentProcessor()
            legend_rows = []
            for label, specs in temp_proc.LABEL_SPECS.items():
                legend_rows.append({"Label": label, "Purpose": specs["Purpose"]})
            st.table(pd.DataFrame(legend_rows))
            idx += 1
            
    # 3. Debug Tab
    with tabs[idx]:
        if 'metrics' in st.session_state and 'debug_images' in st.session_state['metrics']:
            # Get list of available pages from debug_images keys
            avail_pages = sorted(list(st.session_state['metrics']['debug_images'].keys()))
            if avail_pages:
                pg = st.selectbox("Select Page to View Layout", avail_pages)
                
                # Decode the stored bytes back to image
                img_bytes = st.session_state['metrics']['debug_images'][pg]
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert BGR to RGB for Streamlit display
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"PaddleOCR Layout Analysis - Page {pg}", use_column_width=True)
            else:
                st.warning("No debug images captured.")
        else:
            st.info("Run processing to see layout analysis.")
        idx += 1
        
    # 4. AI Tables Tab
    with tabs[idx]:
        if 'ai_table_report_data' in st.session_state:
            for p, tbls in st.session_state['ai_table_report_data'].items():
                st.markdown(f"#### Page {p}")
                for t in tbls:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(base64.b64decode(t['b64_img_ref']), width='stretch')
                    with c2: st.dataframe(t['df'], width="stretch")
        else: st.info("No AI tables.")
        idx += 1
        
    # 5. Downloads Tab
    with tabs[idx]:
        base = os.path.splitext(st.session_state.get('fname', 'doc'))[0]
        
        # Structure Downloads
        st.subheader("1. Structure & Classification Data")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Structure (CSV)", df.to_csv(index=False).encode('utf-8'), f"{base}_structure.csv", "text/csv")
        with c2:
            if 'results' in st.session_state:
                structure_json = []
                for row in st.session_state['results']:
                    keys = ['chunk_id', 'clause_id', 'parent_id', 'level', 'Annex/Appendix', 
                            'content_verbatim', 'references', 'bundle_id', 'Source Page', 
                            'Requirement Label', 'Text Type', 'images', 'tables_html']
                    clean_row = {k: row[k] for k in keys if k in row}
                    structure_json.append(clean_row)
                st.download_button("Download Structure (JSON)", json.dumps(structure_json, indent=2), f"{base}_structure.json", "application/json")
        
        st.markdown("---")
        
        # Executive Summary Downloads
        if summary_data:
            st.subheader("2. Executive Summary Data")
            flat_summary = []
            
            for bid in sorted(summary_data.keys()):
                data = summary_data[bid]
                # Extract meta once
                pg_str = data['page_str']
                
                for item in data['segments']:
                    row = item.copy()
                    # Flatten page/bundle info explicitly into row
                    row['Bundle_ID'] = bid
                    row['Source Page'] = pg_str 
                    flat_summary.append(row)
            
            df_summary = pd.DataFrame(flat_summary)
            # Reorder columns nicely
            cols = ["Bundle_ID", "Source Page", "Requirement Label", "Clause_Range", "Summary_Text", "Key_Quote"]
            # Add any other cols that might exist
            other_cols = [c for c in df_summary.columns if c not in cols]
            final_cols = cols + other_cols
            df_summary = df_summary[[c for c in final_cols if c in df_summary.columns]]

            c5, c6 = st.columns(2)
            with c5:
                st.download_button("Download Summary Report (CSV)", df_summary.to_csv(index=False).encode('utf-8'), f"{base}_summary_report.csv", "text/csv")
            with c6:
                st.download_button("Download Summary Report (JSON)", json.dumps(flat_summary, indent=2), f"{base}_summary_report.json", "application/json")
            st.markdown("---")

        # AI Tables Download
        st.subheader("3. Extracted Tables")
        if 'ai_table_report_data' in st.session_state and st.session_state['ai_table_report_data']:
            html_report = f"<!DOCTYPE html><html><body><h1>Tables - {base}</h1>"
            for p_num in sorted(st.session_state['ai_table_report_data'].keys()):
                for t in st.session_state['ai_table_report_data'][p_num]:
                    html_report += f"<h3>{t['title']} (Page {p_num})</h3>{t['html']}<hr>"
            html_report += "</body></html>"
            st.download_button("📊 Download Tables (HTML)", html_report, f"{base}_tables.html", "text/html")
        else:
            st.button("No AI Tables to Download", disabled=True)