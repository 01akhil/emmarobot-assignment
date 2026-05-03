# Emma Robot Assignment

Screenshot + VLM automation script for:
- locating UI elements (`a`)
- extracting visible text(`b`)
- suggesting next action (`c`)
- scrolling/extracting repeated records (`d`)

## Requirements

- Python 3.10+ (3.11 recommended)
- Google GenAI access/API key (for the purpose of assignment , it is hard-coded, users can change it in case of rpd or rpm issues)
- Desktop permissions for screen capture + mouse control in mac and windows will be asked , thus allow them.

Python packages used:
- `google-genai`
- `pyautogui`
- `Pillow`
- `colorama`
- `pygetwindow` (Windows only; used for window handling)

## Setup

### Windows

1. Open main.py in vs code.(Make sure python is installed)
3. Install dependencies:
   - `pip install google-genai pyautogui pillow colorama pygetwindow`
4. Replace the gemini api key by your generated key in line 32 (API_KEY) of main.py
5. Run:
   - `python main.py`

### macOS

1. Open main.py in vs code.
2. Install dependencies:
   - `python3 -m pip install pyautogui pillow colorama google-genai`
3. Replace the gemini api key by your generated key in line 32 (API_KEY) of main.py   
4. Grant permissions to Terminal/iTerm (and Python) in:
   - **System Settings -> Privacy & Security -> Accessibility**
   - **System Settings -> Privacy & Security -> Screen Recording**
5. Run:
   - `python3 main.py`

## Configuration

The script currently defines API/model settings in `main.py`:
- `API_KEY`
- `LLM_PRIMARY_MODEL`
- `VLM_PRIMARY_MODEL`
- `VLM_OPTION_D_MODEL`
- `RATE_LIMIT_FALLBACK_MODEL`

## Usage Notes

- Keep the terminal visible but not covering the target app area.
- Prompts should be specific and screen-grounded.
- Outputs are saved per mode(in the folder where main.py is kept):
  - `optionA/`, `optionB/`, `optionC/`, `optionD/`

## Option D Caveat

Option D uses `gemini-3.1-flash-lite-preview` (instead of `gemini-3-flash`) due to RPM/RPD constraints.
This helps with quota stability but can reduce bounding-box precision in some views, even when extracted text is correct.

## Troubleshooting

- If model output is empty/malformed, the script now safely falls back or skips invalid data without crashing.
- If screen capture/mouse movement fails:
  - confirm OS permissions
  - avoid running on locked screen
  - keep target content visible and unobstructed
- If you hit quota/rate limits, wait and retry later.
