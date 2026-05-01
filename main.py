# complete code is kept in one script because assignment specifies single script,;
# in production, this would be modularized across multiple files.
# This script implements a screenshot-based assistant that can:
# - find UI elements (Option A),
# - extract on-screen text/regions (Option B),
# - propose next UI action (Option C),
# - scroll and collect table-like rows (Option D).
# It relies on Gemini LLM/VLM calls, screenshot grounding (0-1000 box_2d),
# mouse movement visualization, and structured output logging.
#.env is hardcoded for the purpose of assignment, in production, it would be stored in a .env file.


import os
import time
import json
import sys
import platform
import subprocess
import re
import textwrap
from pathlib import Path
import pyautogui
from PIL import ImageDraw
from colorama import Fore, Style, init
from google import genai

# Initialize colorama for colored terminal output
init(autoreset=True)

# --- Configuration ---
# API/model/runtime settings used across all options.
API_KEY = "AIzaSyBgIHxVLy2Hjhliuf0ssEl-SDxM8SlBrQA" 
CLIENT = genai.Client(api_key=API_KEY)
pyautogui.FAILSAFE = False 
LOG_FILE = "logs/conversation_history.json"
LLM_PRIMARY_MODEL = "gemini-2.5-flash"
VLM_PRIMARY_MODEL = "gemini-3-flash-preview"
# Option D (scroll / table rows) uses a separate vision model because of limited rpm and rpd of above models.
VLM_OPTION_D_MODEL = "gemini-3.1-flash-lite-preview"
# On HTTP 429 / RESOURCE_EXHAUSTED, retry once per attempt with this model before backoff.
RATE_LIMIT_FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"

OUTPUT_ROOT = os.path.dirname(os.path.abspath(__file__))
OPTION_DIRS = {
    "a": os.path.join(OUTPUT_ROOT, "optionA"),
    "b": os.path.join(OUTPUT_ROOT, "optionB"),
    "c": os.path.join(OUTPUT_ROOT, "optionC"),
    "d": os.path.join(OUTPUT_ROOT, "optionD"),
}

# Persistent handle for the assistant terminal window.
# Used so the script can minimize itself before screenshot capture.
ASSISTANT_WINDOW_HANDLE = None

def ensure_option_dirs():
    # Create per-option output folders so saves never fail due to missing dirs.
    for path in OPTION_DIRS.values():
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir_for_file(filepath):

    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)

def update_chat_logs(role, message, goal=None):
    """Saves conversation history to a JSON file for persistence."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "message": message,
        "goal": goal
    }
    
    history = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    
    history.append(log_entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-50:], f, indent=4) 

def get_assistant_handle():
    # Capture active terminal window (Windows) for minimize/restore flow.
    global ASSISTANT_WINDOW_HANDLE
    os_type = platform.system()
    try:
        if os_type == "Windows":
            import pygetwindow as gw
            ASSISTANT_WINDOW_HANDLE = gw.getActiveWindow()
            if ASSISTANT_WINDOW_HANDLE:
               
                pass
    except Exception as e:
        log_event("SYS", f"Warning: Could not capture window handle: {e}")

def _stderr_tty():
    try:
        return sys.stderr.isatty()
    except Exception:
        return False

def format_clickable_path(path):
    """OSC 8 hyperlink so terminals (Cursor, Windows Terminal, etc.) can open the file/folder."""
    abs_p = os.path.abspath(path)
    if not _stderr_tty():
        return abs_p
    try:
        uri = Path(abs_p).as_uri()
    except ValueError:
        return abs_p
 
    return f"\033]8;;{uri}\033\\{abs_p}\033]8;;\033\\"

def log_event(role, message):
    # Centralized logger: colored stderr output + persistent JSON log.
    roles = {
        "SYS": f"{Fore.YELLOW}[SYSTEM]{Style.RESET_ALL}",
        "VLM": f"{Fore.MAGENTA}[VLM]{Style.RESET_ALL}",
        "LLM": f"{Fore.CYAN}[LLM]{Style.RESET_ALL}",
        "ACT": f"{Fore.GREEN}[ACTION]{Style.RESET_ALL}",
        "USER": f"{Fore.BLUE}[USER]{Style.RESET_ALL}",
        "PIX": f"{Fore.WHITE}[SCREEN]{Style.RESET_ALL}",
        "OUT": f"{Fore.GREEN}[OUTPUT]{Style.RESET_ALL}",
    }
  
    print(f"{roles.get(role, f'[{role}]')} {message}", file=sys.stderr, flush=True)
    update_chat_logs(role, message)

def log_step_done(label):
    """Short confirmation that a blocking step finished (visible immediately on stderr)."""
    print(f"{Fore.GREEN}[STEP]{Style.RESET_ALL} Done: {label}", file=sys.stderr, flush=True)

def extract_retry_seconds(error_text):
    """Parse retry delay seconds from Gemini error text when present."""
    patterns = [
        r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
        r"'retryDelay':\s*'([0-9]+)s'",
    ]
    for pat in patterns:
        m = re.search(pat, error_text)
        if m:
            try:
                return max(1, int(float(m.group(1))))
            except ValueError:
                return None
    return None

def classify_api_error(error_text):
    text = error_text.upper()
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "429"
    if "503" in text or "UNAVAILABLE" in text:
        return "503"
    if "401" in text or "UNAUTHENTICATED" in text:
        return "401"
    if "403" in text or "PERMISSION_DENIED" in text:
        return "403"
    return "OTHER"

def friendly_error_message(kind, error_text):
    if kind == "429":
        retry_s = extract_retry_seconds(error_text)
        if retry_s:
            return f"Quota/rate limit reached (429). Suggested retry after about {retry_s}s."
        return "Quota/rate limit reached (429)."
    if kind == "503":
        return "Gemini service temporarily unavailable (503)."
    if kind == "401":
        return "Authentication failed (401). Check API key."
    if kind == "403":
        return "Permission denied (403). Check project access/quota settings."
    short = error_text.strip().splitlines()[0]
    return f"API call failed: {short[:220]}"

def get_primary_model(stream_label):
    # LLM stream is text-only prompt refinement; VLM stream is image-grounded tasks.
    return LLM_PRIMARY_MODEL if stream_label == "LLM" else VLM_PRIMARY_MODEL

def _is_gemini_2_5_or_3_flash_model(model_name):
    """True for primary-tier models that must not be retried after a 429 (2.5 Flash / 3 Flash)."""
    if not model_name:
        return False
    m = model_name.lower().strip()
    if m == RATE_LIMIT_FALLBACK_MODEL.lower().strip():
        return False
    if "3.1" in m:
        return False
    if "2.5" in m and "flash" in m:
        return True
    if "3-flash" in m or "gemini-3-flash" in m:
        return True
    return False

def _generate_with_retries_single_model(contents, stream_label, model_name, retries=5):
    """Up to `retries` attempts on the same model only (with backoff). Used after 429 fallback."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            log_event(
                "SYS",
                f"{stream_label}: attempt {attempt}/{retries} (model={model_name})",
            )
            return CLIENT.models.generate_content(
                model=model_name,
                contents=contents,
            )
        except Exception as e:
            last_error = e
            err_text = str(e)
            kind = classify_api_error(err_text)
            log_event("SYS", f"{stream_label}: {friendly_error_message(kind, err_text)}")
            if attempt < retries:
                wait_s = extract_retry_seconds(err_text) or min(2 ** attempt, 10)
                log_event("SYS", f"Retrying in {wait_s}s...")
                time.sleep(wait_s)
    raise last_error

def generate_with_retries(contents, stream_label, retries=5, model=None):
    """Call Gemini with retries. If model is set, it overrides the default for stream_label.

    On 429 for Gemini 2.5 Flash or 3 Flash, the same model is not retried; we switch to
    RATE_LIMIT_FALLBACK_MODEL and run up to `retries` attempts on that model only.
    """
    # Select explicit model if provided; otherwise use stream default.
    base_model = model or get_primary_model(stream_label)

    if not _is_gemini_2_5_or_3_flash_model(base_model):
        return _generate_with_retries_single_model(
            contents, stream_label, base_model, retries=retries,
        )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            log_event(
                "SYS",
                f"{stream_label}: attempt {attempt}/{retries} (model={base_model})",
            )
            return CLIENT.models.generate_content(
                model=base_model,
                contents=contents,
            )
        except Exception as e:
            last_error = e
            err_text = str(e)
            kind = classify_api_error(err_text)
            if kind == "429":
                log_event(
                    "SYS",
                    f"{stream_label}: 429 — You have 5 RPM and 20 RPD; these have been exhausted, "
                    f"thus falling back to {RATE_LIMIT_FALLBACK_MODEL} (up to {retries} attempts on that model).",
                )
                return _generate_with_retries_single_model(
                    contents, stream_label, RATE_LIMIT_FALLBACK_MODEL, retries=retries,
                )
            log_event("SYS", f"{stream_label}: {friendly_error_message(kind, err_text)}")
            if attempt < retries:
                wait_s = extract_retry_seconds(err_text) or min(2 ** attempt, 10)
                log_event("SYS", f"Retrying in {wait_s}s...")
                time.sleep(wait_s)
    raise last_error

def toggle_window(action="minimize"):
    # Minimize/restore terminal to avoid capturing this script window in screenshots.
    global ASSISTANT_WINDOW_HANDLE
    os_type = platform.system()
    try:
        if os_type == "Windows":
            if ASSISTANT_WINDOW_HANDLE:
                if action == "minimize": ASSISTANT_WINDOW_HANDLE.minimize()
                else: ASSISTANT_WINDOW_HANDLE.restore(); ASSISTANT_WINDOW_HANDLE.activate()
        elif os_type == "Darwin":
            script = 'tell application "Terminal" to activate' if action != "minimize" else 'tell application "System Events" to set visible of process (name of first process whose frontmost is true) to false'
            subprocess.run(['osascript', '-e', script])
        time.sleep(0.5)
    except Exception as e:
        log_event("SYS", f"Window toggle failed: {e}")

def get_scaling_factor(screenshot):
    # Logical screen size (pyautogui) can differ from screenshot pixel size (HiDPI).
    screen_width, screen_height = pyautogui.size()
    physical_width, physical_height = screenshot.size
    return physical_width / screen_width

def box_2d_to_pixel_rect(box_2d, pil_img):
    """Map normalized 0–1000 box_2d [ymin, xmin, ymax, xmax] to pixel rect on the PIL image."""
    w, h = pil_img.size
    ymin, xmin, ymax, xmax = box_2d
    left = int(xmin * w / 1000)
    top = int(ymin * h / 1000)
    right = int(xmax * w / 1000)
    bottom = int(ymax * h / 1000)
    return left, top, right, bottom

def box_2d_center_logical_screen(box_2d, screenshot, scale):
    """Center of box in logical screen coordinates (for pyautogui)."""
    w, h = screenshot.size
    ymin, xmin, ymax, xmax = box_2d
    cx = (xmin + xmax) / 2 * w / 1000
    cy = (ymin + ymax) / 2 * h / 1000
    return cx / scale, cy / scale

def draw_red_boxes(pil_img, boxes_2d, width=2):
    """Return a copy of the image with red rectangles (boxes_2d: list of [ymin,xmin,ymax,xmax])."""
    out = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for box in boxes_2d:
        rect = box_2d_to_pixel_rect(box, pil_img)
        for i in range(width):
            draw.rectangle(
                [rect[0] - i, rect[1] - i, rect[2] + i, rect[3] + i],
                outline=(255, 0, 0),
            )
    return out

def log_screen_capture_info(screenshot, scale):
    sw, sh = pyautogui.size()
    pw, ph = screenshot.size
    log_event(
        "PIX",
        f"Capture size: {pw}x{ph} px (physical). Logical desktop: {sw}x{sh}. "
        f"Scale (physical/logical): {scale:.4f}. Approx. total pixels: {pw * ph:,}.",
    )


def refine_prompt(user_input, task_type):
    # Pre-process user prompt into a stronger instruction before VLM call.
    # This reduces ambiguity and keeps detection/extraction focused on visible UI.
    host_os = platform.system()

    os_hints = {
        "Windows": (
            "Windows UI: taskbar (bottom), Start menu (bottom-left), system tray (bottom-right), "
            "title bar with Minimize/Maximize/Close (top-right), File Explorer ribbon, "
            "Fluent/Win32 controls, context menus with icons."
        ),
        "Darwin": (
            "macOS UI: menu bar (top, full-width), Dock (bottom or side), "
            "traffic-light buttonsClose/Minimize/Zoom (top-left of every window), "
            "Finder sidebar, Spotlight (center screen), native Cocoa controls."
        ),
        "Linux": (
            "Linux UI: varies by desktop — GNOME (top bar, Activities top-left, system tray top-right), "
            "KDE Plasma (taskbar bottom, Application Menu bottom-left), "
            "XFCE/LXDE (panel top or bottom). GTK and Qt widgets. Window buttons position varies."
        ),
    }
    os_context = os_hints.get(host_os, f"{host_os} desktop UI — infer layout from visible elements.")

    # ── Task-specific refinement queries ────────────────────────────────────
    if task_type in ("a", "c"):
        # Finding / acting on a UI element
        refinement_query = f"""You are a prompt engineer for a Vision-Language Model that sees a live screenshot.

TASK: Rewrite the user's goal into a VLM instruction that strictly follows this priority order:

1. SCREEN-FIRST — Describe what to look for ON the visible screen: exact text labels, icon shapes, colors, relative position, exact location,surrounding context. Be specific to what would realistically appear.
2. OS HINTS — {os_context}
3. INTENT — What the user ultimately wants: '{user_input}'

Literal matching constraint (very important):

- Treat '{user_input}' as a LITERAL thing to find on the screen (usually a visible label/text and/or a specific UI element and/or a specific icon at a specific location).
- If the goal references a specific item name (e.g., "option A folder"), the VLM must prioritize finding the element whose visible text/icon matches that name (case-insensitive; allow formatting variants like "optionA", "option-a", "Option A").
- Do NOT reinterpret the goal into a different action/item (example of bad drift: converting "option A folder" into "New folder" or "create folder").

Rules:
- Output ONE concise paragraph. No lists, no headers.
- Do not pad with generic advice. Every sentence must help the VLM locate the element faster.
- If the element could look different per OS, name the {host_os} variant first.

Return ONLY the rewritten VLM instruction."""

    else:
        # Extracting data from screen
        refinement_query = f"""You are an extraction engineer for a Vision-Language Model that sees a live screenshot.

TASK: Rewrite the user's extraction goal into a VLM extraction guide that prioritizes extractable structure on the PRESENT SCREEN before any other assumptions.

The refined guide must strongly emphasize table/grid extraction and multi-section extraction: prioritize identifying table-like grids (headers + rows + columns), list blocks, and repeated card/comment blocks; then identify higher-level page sections (top/header, main content, sidebar, bottom/footer) and extract target information from the correct section(s); if the target appears multiple times across the page, instruct the VLM to extract all relevant occurrences.

The guide should tell the VLM what a correct extraction region (box_2d) represents in the final JSON: coherent table row(s) or cell groups, coherent list block(s), or coherent section block(s) that produce readable text (avoid tiny scattered boxes).

OS/UI context is secondary guidance only: {os_context}.
User extraction intent (literal): '{user_input}'

Output requirements: output ONE concise paragraph with no lists and no headers. No generic advice; every sentence must help the VLM extract the requested information from the screenshot more accurately.

Return ONLY the rewritten VLM extraction guide."""

    try:
        response = generate_with_retries([refinement_query], "LLM")
        text = (response.text or "").strip()
        log_event("LLM", f"Prompt refined (task={task_type}, {(text)} chars).")
        return text
    except Exception as e:
        kind = classify_api_error(str(e))
        log_event("SYS", f"LLM refinement failed: {friendly_error_message(kind, str(e))}. Using raw input.")
        log_event("LLM", f"Prompt refinement fallback (task={task_type}, using raw user input)")
        return user_input

def get_vlm_response(query, pil_img, log_raw=True, model=None):
    try:
        # Supports either one image or multiple images (list/tuple) in a single VLM request.
        payload = [query]
        if isinstance(pil_img, (list, tuple)):
            payload.extend(pil_img)
        else:
            payload.append(pil_img)
        response = generate_with_retries(payload, "VLM", model=model)
        text = (response.text or "").strip()
        if log_raw:
            preview = text if len(text) <= 2000 else text[:2000] + "\n... [truncated for log]"
            log_event("VLM", f"Raw model response:\n{preview}")
        # Model can return wrapper text; extract first JSON object for downstream logic.
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(json_match.group()) if json_match else None
        if parsed is not None:
            pass
        else:
            log_event("SYS", "VLM response had no JSON object; parsed result is None.")
        return parsed
    except Exception as e:
        kind = classify_api_error(str(e))
        log_event("SYS", f"VLM failed after retries: {friendly_error_message(kind, str(e))}")
        return None

def box_2d_to_logical_rect(coords, screenshot, scale):
    """Logical screen rectangle (left, top, right, bottom) for pyautogui from box_2d."""
    ymin, xmin, ymax, xmax = coords
    w, h = screenshot.size
    l_left = (xmin * w / 1000) / scale
    l_top = (ymin * h / 1000) / scale
    l_right = (xmax * w / 1000) / scale
    l_bottom = (ymax * h / 1000) / scale
    if l_right < l_left:
        l_left, l_right = l_right, l_left
    if l_bottom < l_top:
        l_top, l_bottom = l_bottom, l_top
    return l_left, l_top, l_right, l_bottom


def perform_rectangular_mouse_trace(coords, screenshot, scale, include_center=False):
    """Move-only trace: TL -> TR -> BR -> BL -> TL, optionally finish at center."""
    l_left, l_top, l_right, l_bottom = box_2d_to_logical_rect(coords, screenshot, scale)
    sw, sh = pyautogui.size()

    # Keep movement inside current logical desktop bounds.
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    l_left = clamp(l_left, 0, sw - 1)
    l_right = clamp(l_right, 0, sw - 1)
    l_top = clamp(l_top, 0, sh - 1)
    l_bottom = clamp(l_bottom, 0, sh - 1)

    corners = [
        (l_left, l_top),
        (l_right, l_top),
        (l_right, l_bottom),
        (l_left, l_bottom),
        (l_left, l_top),
    ]
    if include_center:
        center_x = (l_left + l_right) / 2
        center_y = (l_top + l_bottom) / 2
        corners.append((center_x, center_y))
    pyautogui.moveTo(corners[0][0], corners[0][1], duration=0.2)
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        dur = max(0.12, min(0.55, 0.0018 * dist))
        pyautogui.moveTo(x1, y1, duration=dur)


def perform_mouse_movement(coords, screenshot, scale):
    # Move only (no click/drag); used for non-click visual highlighting.
    perform_rectangular_mouse_trace(coords, screenshot, scale, include_center=False)

def format_option_b_txt(data):
    """Readable text for extract (option B)."""
    lines = []
    lines.append("Extracted information")
    lines.append("=" * 40)
    if isinstance(data.get("summary"), str) and data["summary"].strip():
        lines.append("")
        lines.append("Summary")
        lines.append("-" * 20)
        lines.append(textwrap.fill(data["summary"].strip(), width=88))
    elements = data.get("elements") or data.get("items") or []
    if not elements:
        lines.append("")
        lines.append("(No structured elements in response.)")
        return "\n".join(lines)
    for i, el in enumerate(elements, 1):
        lines.append("")
        lines.append(f"Item {i}")
        lines.append("-" * 20)
        name = el.get("name") or el.get("label") or f"Region {i}"
        lines.append(f"Label: {name}")
        if el.get("box_2d"):
            lines.append(f"Region (box_2d): {el['box_2d']}")
        content = el.get("extracted_text") or el.get("content") or el.get("text")
        if content:
            lines.append("Content:")
            lines.append(textwrap.fill(str(content), width=88, subsequent_indent="  "))
    return "\n".join(lines)

def format_option_c_txt(data, screenshot, scale):
    action = (data.get("action") or "unknown").lower()
    if action not in ("click", "type", "scroll"):
        action = data.get("action") or "unknown"
    text_val = data.get("text") or data.get("text_to_type") or ""
    reasoning = data.get("reasoning") or ""
    tx = data.get("target_x")
    ty = data.get("target_y")
    if tx is not None and ty is not None:
        cx, cy = float(tx), float(ty)
    elif data.get("box_2d"):
        cx, cy = box_2d_center_logical_screen(data["box_2d"], screenshot, scale)
    else:
        cx, cy = None, None
    lines = [
        "action type: " + str(action),
        f"target coordinates (x, y): ({cx:.1f}, {cy:.1f})" if cx is not None else "target coordinates (x, y): (n/a)",
        "text (if typing is required): " + (str(text_val) if text_val else "(none)"),
        "",
        "reasoning:",
        textwrap.fill(reasoning, width=88) if reasoning else "(none)",
    ]
    return "\n".join(lines)

def parse_user_intent_d(user_goal):
    # Option D supports:
    # - explicit count mode: "extract/get/fetch/collect N records"
    # - open-ended mode: continue until completion cues.
    goal = user_goal.lower()
    count_match = re.search(
        r"\b(?:extract|get|fetch|collect)\s+(\d+)\s+(?:youtube\s+)?(?:comments?|records?|rows?|items?|entries?|results?)\b",
        goal,
    )
    if count_match:
        return {"mode": "COUNT_LIMIT", "limit": int(count_match.group(1))}
    return {"mode": "UNTIL_END", "limit": None}


def get_vlm_data_option_d(previous_screenshot, current_screenshot, user_prompt, collected_count):
    intent = parse_user_intent_d(user_prompt)
    # Option D prompt asks the VLM to do two jobs in one call:
    # 1) extract records from the current frame, and
    # 2) decide if previous vs current frame look identical for stop logic.
    query = f"""
Task: extract readable information from the screen.
You are given two images in order: previous screenshot, then current screenshot.
Use the current screenshot as the primary source for extraction; use the previous screenshot only for temporal context if needed.
Also decide whether both frames are visually the same page state (same table content/viewport, ignoring tiny rendering noise).
For extraction regions, return coherent extract units only:
- in tables/grids, return complete row blocks (or coherent row-level cell groups), not tiny fragments;
- in lists/cards/sections, return complete item or section blocks;
- when target data repeats, extract all visible matching units in the current screenshot.
Each "box_2d" must tightly cover the same visual unit described by its "extracted_text".
User goal: {user_prompt}
Already extracted unique records: {collected_count}
Intent mode: {intent["mode"]}; limit: {intent["limit"] if intent["limit"] is not None else "N/A"}

Return JSON only:
{{
  "summary": "...",
  "reasoning": "why this status was selected from visible evidence",
  "visual_cues": {{
    "frames_identical": true or false,
    "scroll_bar_at_end": true or false,
    "has_more_scrollable_content": true or false,
    "empty_sheet_or_document": true or false,
    "has_required_data_in_view": true or false
  }},
  "elements": [
    {{
      "name": "...",
      "box_2d": [ymin, xmin, ymax, xmax],
      "extracted_text": "..."
    }}
  ]
}}
"""
    data = get_vlm_response(
        query,
        [previous_screenshot, current_screenshot],
        log_raw=False,
        model=VLM_OPTION_D_MODEL,
    )
    if data is None:
        log_event("SYS", "Option D: no valid JSON from VLM for this iteration.")
    return data


def refine_option_d_box_for_text(current_screenshot, user_goal, extracted_text, proposed_box):
    """Run a focused grounding pass so box and extracted_text refer to the same row."""
    if not extracted_text:
        return proposed_box
    query = f"""
Task: locate the exact visual row/item on the CURRENT screenshot that matches this extracted row text.
User goal: {user_goal}
Target extracted_text (literal): {extracted_text}
Proposed box_2d from previous pass: {proposed_box}

Rules:
- Match the literal row/item text first (date/amount/description/reference/status where visible).
- Return one tight row-level/coherent item-level box_2d for that same row/item.
- If uncertain, prefer the row containing the literal target text over nearby similar rows.

Return JSON only:
{{
  "box_2d": [ymin, xmin, ymax, xmax]
}}
"""
    data = get_vlm_response(query, current_screenshot, log_raw=False, model=VLM_OPTION_D_MODEL)
    if isinstance(data, dict):
        b = data.get("box_2d")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                return list(map(float, b))
            except Exception:
                return proposed_box
    return proposed_box


def run_option_d(user_prompt):
    # Option D loop:
    # capture -> ask VLM (previous+current) -> collect unique rows -> highlight/save -> scroll.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OPTION_DIRS["d"]
    ensure_option_dirs()
    os.makedirs(out_dir, exist_ok=True)

    intent = parse_user_intent_d(user_prompt)
    limit = intent["limit"]
    all_elements = []    # final deduplicated records to write into output TXT
    seen_texts = set()   # dedupe key: extracted_text
    previous_screenshot = None
    last_summary = "N/A"
    iteration = 0
    empty_view_streak = 0

    while True:
        iteration += 1
        log_event("SYS", f"Option D: scanning page (Iteration {iteration})...")
        screenshot = pyautogui.screenshot()
        scale = get_scaling_factor(screenshot)

        # Always send two screenshots (previous + current).
        # On iteration 1, previous does not exist, so we duplicate current to keep
        # a consistent 2-image interface for the VLM.
        previous_for_vlm = previous_screenshot if previous_screenshot is not None else screenshot
        # Send both frames each iteration so VLM can judge whether viewport changed.
        data = get_vlm_data_option_d(previous_for_vlm, screenshot, user_prompt, len(all_elements))
        if not data:
            log_event("SYS", "Option D: stopping due to missing VLM data.")
            break

        last_summary = data.get("summary", last_summary)
        cues = data.get("visual_cues") or {}
        log_event("VLM", f"Option D reasoning: {data.get('reasoning', 'No reasoning provided.')}; cues={json.dumps(cues)}")

        # VLM decides whether previous and current frames are identical.
        # We ignore this cue on the first loop (no real previous frame yet).
        if previous_screenshot is not None and bool(cues.get("frames_identical")):
            log_event("OUT", "Option D: completion criteria met (VLM says last two frames are identical).")
            break

        boxes = []  # coordinates for visual highlight + optional PNG boxing
        added_this_iteration = 0
        for el in data.get("elements", []):
            txt = (el.get("extracted_text") or "").strip()
            # De-duplicate by extracted text so repeated rows on nearby scrolls
            # are not written multiple times.
            if txt and txt not in seen_texts:
                b = el.get("box_2d")
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    b = list(map(float, b))
                    # Second-pass grounding: enforce alignment between row text and box.
                    refined_box = refine_option_d_box_for_text(screenshot, user_prompt, txt, b)
                    el["box_2d"] = refined_box
                    boxes.append(refined_box)
                seen_texts.add(txt)
                all_elements.append(el)
                added_this_iteration += 1
                if limit and len(all_elements) >= limit:
                    log_event("OUT", f"Option D: reached requested limit of {limit}.")
                    break

        has_required_data = bool(cues.get("has_required_data_in_view"))
        is_empty_view = bool(cues.get("empty_sheet_or_document"))
        if is_empty_view:
            empty_view_streak += 1
            log_event("SYS", f"Option D: empty sheet/document detected ({empty_view_streak}/2).")
            if empty_view_streak >= 2:
                log_event("OUT", "Option D: completion criteria met (empty view detected twice).")
                break
        else:
            empty_view_streak = 0

        # User-requested behavior: when no new records are found, do not draw
        # or move across boxes; just continue the scroll loop.
        if added_this_iteration == 0:
            log_event("SYS", "Option D: no new records in this view; skipping box draw/highlight and scrolling.")
        elif boxes:
            for idx, box in enumerate(boxes, 1):
                log_event("ACT", f"Option D: highlighting record {idx}/{len(boxes)} — box_2d={box}")
                perform_mouse_movement(box, screenshot, scale)
                time.sleep(0.35)

        # Don't save image when sheet/doc is empty or there is no required data in this view.
        should_save_image = bool(boxes) and has_required_data and (added_this_iteration > 0)
        if should_save_image:
            img = draw_red_boxes(screenshot, boxes)
            png_path = os.path.join(out_dir, f"extract_{stamp}_iter{iteration}.png")
            ensure_parent_dir_for_file(png_path)
            img.save(png_path)
            log_event("OUT", f"Option D output PNG — {format_clickable_path(png_path)}")
        else:
            log_event("SYS", "Option D: no required data in this view (or empty view); skipping PNG save.")

        # Hard stop when user requested an exact count and we've reached it.
        if limit and len(all_elements) >= limit:
            break

        # Stop when VLM indicates bottom/end of scrollable content.
        if bool(cues.get("scroll_bar_at_end")):
            log_event("OUT", "Option D: completion criteria met (scrollbar at end).")
            break

        # Update previous frame before next scroll so the next iteration compares
        # the new screenshot against this one.
        previous_screenshot = screenshot
        pyautogui.scroll(-800)
        time.sleep(1.5)

    # Final consolidated text output for all collected records.
    txt_path = os.path.join(out_dir, f"output_{stamp}.txt")
    ensure_parent_dir_for_file(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Summary: {last_summary}\n\n")
        for el in all_elements:
            f.write(f"--- {el.get('name', 'record')} ---\n{el.get('extracted_text', '')}\n\n")
    log_event("OUT", f"Option D output TXT — {format_clickable_path(txt_path)}")
    log_event("OUT", f"Option D output folder — {format_clickable_path(out_dir)}")

def main():
    # Interactive loop:
    # - choose mode
    # - capture screenshot
    # - run mode-specific VLM logic
    # - save artifacts into option folders
    get_assistant_handle()
    ensure_option_dirs()
    

    while True:
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== Emma Robot Assignment ==={Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== Instructions ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}1.This is a screenshot+VLM based script which helps user in performing below given tasks using user prompts. Thus allow terminal to take screenshots and permission to interact.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}2.Specific and detailed prompt helps VLM in not getting confused between similar elements{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}3.Most of the tasks do not have scroll functionality, thus give it instructions to work on the visible pixels.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}4. Users are advised to keep this terminal along side  or on the main application in a small screen such that , it does not obstruct the view of the main application and let users see the logs in terminal about processing.{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}5. The terminal will minimize before taking screenshot, and comes back after that.If it doesnt, user can restore it manually. Users can checkout demo videos here https://drive.google.com/drive/folders/1NadI8j2AZd9NSA_7f3KQcnSYocrdp8WG?usp=sharing{Style.RESET_ALL} \n")

        
        choice_menu = (
            f"{Fore.BLUE}Choice:{Style.RESET_ALL}\n"
            f"  a) Find the (x, y) coordinates/pixels of a described UI element\n"
            f"  b) Extract text from the screen (based only on the visible pixels)\n"
            f"  c) Determine the next action based on the current screen description\n"
            f"  d) Scroll and find the (x, y) coordinates of row(s) in a table\n"
            f"  q) Quit\n"
        )
        choice = input(choice_menu + f"{Fore.BLUE}Your choice (a/b/c/d or q): {Style.RESET_ALL}").lower().strip()
        
        if choice == 'q': break
        if choice not in ['a', 'b', 'c', 'd']: continue

        # User prompt examples depend on mode (a/b/c).
        example = ""
        if choice == "a":
            example = "locate the share element on spreadsheet"
        elif choice == "b":
            example = "extract comments from this youtube video"
        elif choice == "d":
            example = "get all rows where description is coffee"
        else:
            example = "request edit access for this document"

        user_prompt = input(
            f"\n{Fore.BLUE}Please write your detailed prompt (eg. {example}): {Style.RESET_ALL}",
        )
        update_chat_logs("USER", user_prompt, goal=choice)

        # Option D has its own internal screenshot+scroll loop.
        if choice == "d":
            run_option_d(user_prompt)
            print(f"\n{Fore.WHITE}{'—'*30}{Style.RESET_ALL}")
            continue

        # For A/B/C, capture once, then run one-shot analysis on visible frame.
        log_event("SYS", "Minimizing window for capturing screenshot…")
        toggle_window("minimize")
        screenshot = pyautogui.screenshot()
        log_event("PIX", "Screenshot captured.")
        scale = get_scaling_factor(screenshot)
        toggle_window("restore")
        log_event("SYS", "Restored window after screenshot capture.")

        log_screen_capture_info(screenshot, scale)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OPTION_DIRS[choice]
        ensure_option_dirs()
        os.makedirs(out_dir, exist_ok=True)

        log_event("SYS", "Step: refining prompt…")
        refined_strategy = refine_prompt(user_prompt, choice)
        
        if choice == 'a':
            # Option A: locate one UI element and trace its rectangle with mouse.
            query = (
                f"Find the UI element whose visible label/text best matches the literal goal: {user_prompt!r}. "
                f"Use this VLM instruction to guide detection: {refined_strategy}. "
                "The screenshot may be Windows, macOS, or Linux; locate the best match. "
                "Return JSON only: "
                '{"box_2d": [ymin, xmin, ymax, xmax]} with values 0-1000.'
            )
            log_event("SYS", "Step: passing screenshot and prompt to VLM…")
            data = get_vlm_response(query, screenshot)
            
            if data and 'box_2d' in data:
                log_event("ACT", f"Found element at box_2d={data['box_2d']}")
                cx, cy = box_2d_center_logical_screen(data['box_2d'], screenshot, scale)
                log_event("PIX", f"Target center (logical screen): ({cx:.1f}, {cy:.1f})")
                log_event("SYS", "Step: moving mouse along the found element rectangle…")
                perform_rectangular_mouse_trace(data['box_2d'], screenshot, scale, include_center=True)
                log_step_done("moving mouse (option A)")
                log_event("SYS", "Step: saving output PNG with annotated rectangles…")
                img = draw_red_boxes(screenshot, [data['box_2d']])
                png_path = os.path.join(out_dir, f"find_{stamp}.png")
                ensure_parent_dir_for_file(png_path)
                img.save(png_path)
                log_event("OUT", f"Output PNG — {format_clickable_path(png_path)}")
                log_event("OUT", f"Output folder — {format_clickable_path(out_dir)}")
                log_step_done("save PNG (option A)")
            else:
                log_event("SYS", "Option A: no valid box_2d; nothing saved.")

        elif choice == 'b':
            # Option B: extract structured text regions and optionally highlight all boxes.
            query = (
                f"Task: extract readable information from the screen. User goal: {user_prompt!r}. "
                f"Use this extraction guide: {refined_strategy}. "
                "Focus first on table/grid structures (headers + rows + cells) and on multi-section pages (header/main/sidebar/footer); also extract repeated blocks (cards/lists/comments) when relevant. "
                "For box_2d regions, return coherent extract units (table rows/cell groups, list blocks, or section blocks), not tiny fragments. "
                "Identify distinct UI regions and return JSON only:\n"
                '{"summary": "short overall summary", "elements": ['
                '{"name": "short label", "box_2d": [ymin, xmin, ymax, xmax], "extracted_text": "text from that region"}'
                "]}\n"
                "Use box_2d in 0-1000 normalized coordinates like Gemini UI grounding."
            )
            log_event("SYS", "Step: passing screenshot and prompt to VLM…")
            data = get_vlm_response(query, screenshot)
            log_step_done("got this from VLM")
            boxes = []
            if data:
                for el in data.get("elements") or data.get("items") or []:
                    b = el.get("box_2d")
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        boxes.append(list(map(float, b)))
            if boxes:
                log_event("SYS", "Step: moving mouse across each extracted region on screen…")
                for idx, box in enumerate(boxes, 1):
                    log_event("ACT", f"Highlighting extract item {idx}/{len(boxes)} — box_2d={box}")
                    perform_mouse_movement(box, screenshot, scale)
                    time.sleep(0.4)
                log_step_done("moving mouse (option B, all regions)")

            log_event("SYS", "Step: saving output PNG with annotated rectangles…")
            if boxes:
                img = draw_red_boxes(screenshot, boxes)
                png_path = os.path.join(out_dir, f"extract_{stamp}.png")
                ensure_parent_dir_for_file(png_path)
                img.save(png_path)
                log_event("OUT", f"Output PNG — {format_clickable_path(png_path)}")
            else:
                log_event("SYS", "Option B: no boxes to draw; saving screenshot without boxes.")
                png_path = os.path.join(out_dir, f"extract_{stamp}.png")
                ensure_parent_dir_for_file(png_path)
                screenshot.convert("RGB").save(png_path)
                log_event("OUT", f"Output PNG — {format_clickable_path(png_path)}")
            log_step_done("save PNG (option B)")

            txt_path = os.path.join(out_dir, f"output_{stamp}.txt")
            body = format_option_b_txt(data) if data else "No parsed data from VLM."
            log_event("SYS", "Step: writing output.txt…")
            ensure_parent_dir_for_file(txt_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(body)
            log_event("OUT", f"Output TXT — {format_clickable_path(txt_path)}")
            log_event("OUT", f"Output folder — {format_clickable_path(out_dir)}")
            log_step_done("save output.txt (option B)")

        elif choice == 'c':
            # Option C: ask VLM for next action plan (click/type/scroll) + rationale.
            query = (
                f"Goal: {refined_strategy}. The UI may be Windows, macOS, or Linux. "
                "Analyze the screen and decide the next step. "
                "Return JSON only with keys: "
                '"action" (one of: click, type, scroll), '
                '"box_2d": [ymin, xmin, ymax, xmax] in 0-1000 if the action targets a region (else null), '
                '"text": string to type if action is type (else empty string), '
                '"reasoning": brief explanation.'
            )
            log_event("SYS", "Step: passing screenshot and prompt to VLM…")
            data = get_vlm_response(query, screenshot)
            log_step_done("got this from VLM")
            if data:
                action = (data.get("action") or "").lower()
                log_event("VLM", f"Planned action: {action}; reasoning: {data.get('reasoning', '')[:500]}")
                if data.get("box_2d"):
                    cx, cy = box_2d_center_logical_screen(data["box_2d"], screenshot, scale)
                    log_event("PIX", f"Target center (logical screen): ({cx:.1f}, {cy:.1f})")
                    log_event("SYS", "Step: moving mouse along the planned rectangle…")
                    perform_rectangular_mouse_trace(data['box_2d'], screenshot, scale, include_center=True)
                    log_step_done("moving mouse (option C)")

                log_event("SYS", "Step: saving output PNG with annotated rectangles…")
                boxes_c = []
                if data.get("box_2d"):
                    boxes_c.append(data["box_2d"])
                if boxes_c:
                    img = draw_red_boxes(screenshot, boxes_c)
                else:
                    img = screenshot.copy().convert("RGB")
                png_path = os.path.join(out_dir, f"logic_{stamp}.png")
                ensure_parent_dir_for_file(png_path)
                img.save(png_path)
                log_event("OUT", f"Output PNG — {format_clickable_path(png_path)}")
                txt_path = os.path.join(out_dir, f"output_{stamp}.txt")
                ensure_parent_dir_for_file(txt_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(format_option_c_txt(data, screenshot, scale))
                log_event("OUT", f"Output TXT — {format_clickable_path(txt_path)}")
                log_event("OUT", f"Output folder — {format_clickable_path(out_dir)}")
                log_step_done("save PNG + output.txt (option C)")
            else:
                log_event("SYS", "Option C: no parsed data; no files saved.")

        print(f"\n{Fore.WHITE}{'—'*30}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()


