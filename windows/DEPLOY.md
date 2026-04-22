# Merlin — Windows Deployment Runbook

*Written for Claude Code / Claude Desktop running on the user's Windows laptop to execute alongside the user.*

## Context

You are helping install Merlin on a Windows laptop. Merlin is an ambient AI desk companion — camera (EMEET PIXY) + mic + local LLM + TTS. All local, no cloud. The user is sitting at the machine with you.

**Assumption:** The user does not know Windows well. Be explicit about keystrokes (Windows key, right-click, etc.) and confirm each phase before moving on.

**Hardware expected:**
- Windows 11 laptop
- NVIDIA GPU (GTX 1060 6GB minimum, RTX 4060 recommended). If no NVIDIA GPU: note it — we'll switch to CPU mode in config.
- EMEET PIXY plugged in via USB
- Speaker (built-in or Bluetooth)

**Source of truth:** `rebel-builder/merlin-bot` repo on GitHub, `windows/` folder. Public repo.

---

## The 5 Phases

1. **Prerequisites** — Python, Git, Node, LM Studio via `winget`
2. **Claude Code CLI install** — so Claude can drive the rest
3. **Clone merlin-bot** — git clone into `C:\merlin`
4. **Run `install.ps1`** — venv, pip, models, device checks
5. **Gemma in LM Studio + start Merlin** — voice loop live

Do each phase in order. Verify success before continuing. If something errors, stop and debug — don't skip.

---

## Phase 1 — Windows prerequisites

The user must run these. Claude can't yet because Claude Code isn't installed.

**Tell the user:**

> Press the Windows key, type `powershell`, right-click **Windows PowerShell**, and choose **Run as administrator**. Click Yes on the UAC prompt. Then paste this block and press Enter:

```powershell
winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
winget install --id Git.Git -e --accept-source-agreements --accept-package-agreements
winget install --id OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
winget install --id LMStudio.LMStudio -e --accept-source-agreements --accept-package-agreements
```

Takes 5–10 min.

**Verify (in a NEW, non-admin PowerShell window so PATH refreshes):**

```powershell
python --version
git --version
node --version
```

All three should print a version. If any says "not recognized," the PATH didn't pick up — ask the user to restart PowerShell, or in rare cases restart the machine.

**Checkpoint:** All four installed, all three version checks pass → go to Phase 2.

---

## Phase 2 — Install Claude Code

Still in regular (non-admin) PowerShell:

```powershell
npm install -g @anthropic-ai/claude-code
```

Wait for completion. Then verify:

```powershell
claude --version
```

Should print a Claude Code version.

**Login:**

```powershell
claude
```

This prompts for authentication. The user logs in with their Anthropic account (browser opens). Once logged in, exit Claude Code with `/exit` or Ctrl+C.

**Checkpoint:** `claude --version` works, login complete → go to Phase 3.

---

## Phase 3 — Clone the repo

```powershell
cd C:\
git clone https://github.com/rebel-builder/merlin-bot.git merlin
cd merlin\windows
```

The project now lives at `C:\merlin\windows\`. Verify:

```powershell
dir
```

Should show `merlin.py`, `setup.bat`, `requirements.txt`, etc.

**Now open Claude Code in this folder:**

```powershell
claude
```

From here, Claude Code drives. Claude: you are now in `C:\merlin\windows\`. Read `CLAUDE.md` and `README.md` in this folder if they exist. If they're missing (older repo state), the key files to know are:

- `merlin.py` — entry point
- `config.py` — all settings
- `install.ps1` — installer (may not exist in repo yet — if missing, run `setup.bat` instead)
- `setup.bat` — fallback installer

**Checkpoint:** Claude Code is running inside `C:\merlin\windows\` → go to Phase 4.

---

## Phase 4 — Run the installer

**If `install.ps1` exists in the folder**, it's the better installer (has validation + summary). Run it from PowerShell (not Claude Code — it's interactive):

```powershell
# In PowerShell, in C:\merlin\windows
Set-ExecutionPolicy -Scope Process Bypass
.\install.ps1
```

**If only `setup.bat` exists**, double-click it in Explorer, or run:

```powershell
.\setup.bat
```

Either way, the installer will:
1. Create `.venv` (or `venv`) virtual environment
2. Install Python packages from `requirements.txt` (5–10 min)
3. Download YuNet face detection model (~300 KB)
4. Check / prompt for Kokoro TTS models (see below)
5. Check LM Studio is reachable
6. List cameras and audio devices

**Kokoro models must be downloaded manually.** Two files, ~330 MB total:

```powershell
# Still in C:\merlin\windows
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" -OutFile "kokoro-v1.0.onnx"
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "voices-v1.0.bin"
```

**Common Phase 4 errors (from Josiah's deploy):**

| Error | Fix |
|---|---|
| `SSL certificate error` on YuNet download | Download manually from the URL in `README.md`, save as `face_detection_yunet_2023mar.onnx` in `C:\merlin\windows\` |
| `Library cublas64_12.dll is not found` | `pip install nvidia-cublas-cu12` (inside the venv — activate it first: `.\venv\Scripts\activate`) |
| `No module named 'sounddevice'` or similar | venv not activated. Run `.\venv\Scripts\activate`, look for `(venv)` in the prompt |
| `Set-ExecutionPolicy` blocked | Ran from wrong scope. Use `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (one-time) |
| No NVIDIA GPU / Whisper CUDA errors | Edit `config.py`: `WHISPER_DEVICE = "cpu"` and `WHISPER_COMPUTE = "int8"` |
| Smart App Control blocks `merlin.py` | Windows Settings → Privacy & Security → Smart App Control → turn off |
| Camera not found (wrong index) | Edit `config.py`: try `CAMERA_INDEX = 0`, then `1`, then `2` until PIXY is detected |

**Checkpoint:** Installer reports all PASS (or only Kokoro/LM Studio failures, which we'll fix in Phase 5) → go to Phase 5.

---

## Phase 5 — LM Studio + launch Merlin

**5a. Load a model in LM Studio:**

Tell the user:
> Open LM Studio from the Start menu. Go to the search/download tab, search for `gemma-4-4b-it` and download it. (If a newer Gemma like `gemma-4-e4b-it` is available and smaller, that works too.) Then:
> 1. Click the **Local Server** tab (left sidebar)
> 2. Select the Gemma model at the top
> 3. Click **Start Server**
> 4. Confirm the port is 1234 (default)

Verify from PowerShell:

```powershell
curl http://localhost:1234/v1/models
```

Should return JSON listing the loaded model.

**5b. Start Merlin:**

```powershell
cd C:\merlin\windows
.\venv\Scripts\activate
python merlin.py
```

Expected boot sequence (first run is slow — ~20–30 sec):
1. `audio.py` opens mic
2. `stt.py` downloads Whisper Small model (~150 MB, only first time)
3. `voice.py` loads Kokoro
4. `tracker.py` opens camera, detects face
5. Console prints something like `Listening...`

**5c. Test the voice loop:**

User says: **"Hey Merlin"**

Merlin should:
- Play a listening tone
- After user speaks, play a thinking tone
- Respond out loud via speaker

**Common Phase 5 errors:**

| Symptom | Fix |
|---|---|
| Merlin says nothing | Check speaker is Windows default output. Run `python -c "import sounddevice; print(sounddevice.query_devices())"` and set `SPEAKER_DEVICE` in `config.py` by index |
| Merlin speaks but picks up own voice (echo loop) | Raise `ENERGY_THRESHOLD` in `config.py` to `0.04` or `0.06`. Move mic away from speaker |
| Merlin ignores wake word | Check mic is the PIXY, not laptop internal. Set `MIC_DEVICE` in `config.py` by index |
| Merlin speaks `<think>...</think>` out loud | Model is emitting reasoning tokens. `brain.py` strips them but double-check a non-reasoning model is loaded. Gemma 4 4B/E4B is fine |
| First response takes 30+ sec | Model too big / thinking too hard. Switch to smaller model in LM Studio |

**Checkpoint:** User says "Hey Merlin," Merlin responds out loud. Done.

---

## Done-checks

Before declaring victory:

- [ ] User said "Hey Merlin" and got a spoken response
- [ ] Camera LED is on / face tracking working (if PIXY supports PTZ, it's moving)
- [ ] No errors in Merlin's console
- [ ] Mute test: user says "Stop listening" → Merlin stops. User says "Wake up" → Merlin resumes.
- [ ] Desktop shortcut created for `merlin.py` (optional but nice)

## Final hand-off

Tell the user:
- Wake word is **"Hey Merlin"** (or just "Merlin")
- After Merlin answers, 30-second window to keep talking without re-saying the wake word
- Say **"Stop listening"** to mute, **"Wake up"** to resume
- To restart Merlin after a reboot: open PowerShell, `cd C:\merlin\windows`, `.\venv\Scripts\activate`, `python merlin.py`
- LM Studio must be running with the model loaded before Merlin starts

---

## If things go badly wrong

- Full reset: delete `C:\merlin` and re-clone
- Delete just `.venv` (or `venv`) and re-run installer
- `pip list` inside the venv to see what's actually installed
- Logs: Merlin prints to the PowerShell window — copy/paste errors here so Claude can read them

Grant is the canonical deployment owner (see `CLAUDE.md`). If this deploy is going sideways and the user is getting frustrated, pause and ping Grant rather than pushing through.
