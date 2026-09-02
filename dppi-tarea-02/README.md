# Album Cover Gesture Detector

Point your webcam at yourself, make a face/hand gesture, and display the corresponding iconic album cover in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM).

Two windows/panes side by side: 
- **Camera** — your webcam feed with landmarks drawn on top, plus a live debug readout (gesture, yaw, pitch)
- **Meme / Album** — the album cover matching whatever gesture you're currently making

## Gestures

| # | Gesture | How to trigger | Album Cover |
|---|---|---|---|
| 1 | Cruz / Cross | Cruzar un dedo horizontal con otro vertical | `BLIZZARD_OZZY.webp` (Ozzy Osbourne) |
| 2 | Dos manos en la cara | Dos manos tapan la cara | `BOOTLEG_DYLAN.jpg` (Bob Dylan) |
| 3 | Heroes Pose | Una mano bajo la cara y otra en el aire al costado | `HEROES_BOWIE.jpg` (David Bowie) |
| 4 | Manos a los hombros | Dos manos extendidas cruzadas tocando los hombros | `QUEEN2_QUEEN.jpeg` (Queen II) |
| 5 | Cabeza atrás (perfil) | Cabeza inclinada hacia atrás de perfil (yaw + pitch) | `MADONNA_TRUE_BLUE.jpeg` (Madonna) |
| 6 | Cabeza atrás (frente) | Cabeza inclinada hacia atrás de frente (pitch) | `THEBENDS_RADIOHEAD.jpeg` (Radiohead) |
| - | Default / Reposo | Sin gesto activo (estado de espera) | `rick_cover.jpg` (Rick Astley) |

## Running it — desktop (Python)

Requires Python 3 and a webcam.

Easiest way: just double-click **`Launch Gesture Meme.command`**. First run takes a minute to set itself up (installs everything automatically), then launches straight away. Every run after that is instant.

**First time opening it:** macOS will warn "cannot be opened because it is from an unidentified developer" — this is normal for any downloaded script, not specific to this one. Right-click the file → **Open** → click **Open** in the dialog that appears. You only need to do this once.

Or manually, if you prefer Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Press `q` or `Esc` in the Camera window to quit.

## Running it — browser

No install needed, but the webcam API requires serving over HTTP (opening `index.html` directly as a `file://` URL will not get camera permission). From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` and allow camera access. Models load from Google's hosted MediaPipe CDN at runtime, so nothing local is needed for the browser version.

## Live debug HUD

The Camera window always shows a small readout in the top-left corner:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
```

Useful for tuning the detection thresholds at the top of `gesture_meme.py` / `app.js` if a gesture is triggering too easily or not easily enough for your setup/lighting.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM)
index.html        browser UI shell
memes/            meme images (+ one video, unused for now)
models/           MediaPipe .task model files used by the desktop version
requirements.txt  Python dependencies
```
