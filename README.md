# Song Eternal

A small Python app that analyzes a local audio file and extends playback using
Eternal Jukebox-style beat jumps.

EternalJukebox reference project: https://github.com/EternalBox/EternalJukebox

Song Eternal can be used as a song extender, MP3 extender, or music loop
generator for making a track play much longer while still following the
original song structure.

The app detects beats, extracts per-beat audio features, finds similar beats,
and streams the song beat by beat. As playback continues, it occasionally jumps
to a similar beat instead of continuing linearly. If playback eventually reaches
the final beat, it stops.

## Setup

Requirements:

- Python 3.10 or newer
- an audio output device
- `ffmpeg` only if you want to export extended MP3 files

From the project folder, create a virtual environment and install the Python
packages:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you already have a working `.venv`, you only need the final install command.

## Run

Start the GUI with:

```powershell
.\.venv\Scripts\python.exe main.py
```

If you are not using a virtual environment, install the packages from
`requirements.txt` into your active Python environment, then run:

```powershell
python main.py
```

The GUI lets you open an audio file, analyze it, play/stop playback, export an
extended MP3, and watch the waveform, beat grid, branch map, playhead, and
latest jump.

Timed ending is optional. When enabled, branches remain active for the chosen
hours/minutes, then branching is disabled and playback continues normally to
the end of the song.

If there is an `.mp3` file in the same folder as `main.py`, it is selected
automatically when the GUI opens.

## Files Needed To Run

The app only needs these project files:

- `main.py`
- `requirements.txt`

Optional local files:

- audio files such as `.mp3`, `.wav`, `.flac`, `.ogg`, or `.m4a`
- `ffmpeg.exe`, if you want MP3 export without installing `ffmpeg` on PATH

## Export MP3

Click `Export MP3` after analysis. The app opens a separate export-length
window where you choose hours, minutes, and seconds, then asks where to save the
MP3. Long exports are streamed through `ffmpeg`, so a 10-hour export does not
need to be held in memory.

The app looks for `ffmpeg.exe` next to `main.py`, then on PATH. If it cannot
find one, it asks you to choose `ffmpeg.exe`.

The default settings are tuned for more natural jumps:

- backwards jumps
- long jumps
- same-bar-position jumps
- similar beat volume levels
- about 10 total branches per minute by default
- less-used branches are favored during playback
- branch usage counts normalize once every route has enough shared usage
- branch chance ramps to 100% near the final available branch points
- nearby branches with the same jump offset are collapsed into one route
- hard beat-boundary jumps with no crossfade
- a global branch budget that keeps only the best matches

All tuning is done in the GUI.

## Credits

This project is inspired by EternalJukebox, a rehosting/fork of the Infinite
Jukebox concept, and its approach to looping songs by finding musically similar
beat-to-beat jumps.

This Python version was created with ChatGPT. Code from EternalJukebox was
provided to ChatGPT as reference material while developing this version.

EternalJukebox is licensed under the MIT License. See
`THIRD_PARTY_NOTICES.md` for the EternalJukebox copyright and license notice.
