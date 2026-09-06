# Guided Mode

The robot looks around, tells you what it sees, proposes what to do next and
asks you before doing it — out loud. It builds a map underneath as it goes.

```
look  →  "I see a hallway with a chair on the left."
think →  "I'd like to turn left because there is more open floor that way. Is that a good choice?"
you   →  yes | no | left | right | forward | back | stay | quit
act   →  the robot moves (after a physical obstacle check), then looks again
```

If you say **no**, it proposes something else (up to `max_reproposals` times).
If you give an **instruction** ("no, turn right"), it does that instead.
If you say nothing for `answer_timeout_s`, it stays put and looks again.
**Forward is always checked against the distance sensor and depth map first.**
The AI's opinion never overrides the sensors.

## Run it

```bash
source ~/picrawler-ai/.venv/bin/activate
cd ~/picrawler-ai/picrawler-ai-4
python main.py --mode guided --duration 10
```

Answer by typing and pressing Enter, or by voice if a microphone is plugged in
(see below). The robot speaks through the speaker.

## How much it asks (`--autonomy`)

| Mode | Behaviour |
|------|-----------|
| `ask_always` (default) | Asks before every action |
| `ask_when_unsure` | Asks only when the AI's confidence is below `confidence_threshold` (0.7) or the decision came from sensor rules rather than the AI |
| `ask_forward_only` | Asks before walking forward; turns and backing up are just narrated |
| `never_ask` | Narrates and acts on its own |

```bash
python main.py --mode guided --autonomy ask_forward_only --duration 10
```

In any autonomous mode you can still type while it works: `stop` or `no` makes
it pause and ask, `left` / `right` / `forward` / `back` overrides the next move,
`quit` ends the session. The setting also lives in `guided_settings.autonomy`.

## Answering by voice

Plug a USB microphone into the Pi and check it:

```bash
python utils/check_hardware.py        # section 7 lists capture devices
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav && aplay test.wav
```

`listen` defaults to `auto`: if a capture device is found and an OpenAI key is
set, the robot records `listen_seconds` after each question and transcribes it
with OpenAI. If it hears nothing, or recording fails, it falls back to the
keyboard for that answer. Force one or the other with `--listen keyboard` or
`--listen microphone`.

## Mapping while you steer

Guided mode runs the SLAM stack underneath (`build_map`, default on) and streams
the map over UDP exactly like `slam_explore`. Watch it live on a laptop:

```bash
python tools/map_viewer.py            # laptop, same Wi-Fi as the robot
```

The 2D map is saved to `logs/guided_map_final.jpg` when the session ends.
Use `--no-map` to skip mapping on a slow Pi.

## Requirements

- An OpenAI key. Either put it in `config/config.json` (`openai_api_key`) or,
  cleaner, export it so `git pull` never fights with your local edit:

  ```bash
  echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
  source ~/.bashrc
  ```

- A speaker (for `ffplay`, `mpg123` or `aplay`). Without one the robot's
  speech is printed in the log instead.

Without a key, guided mode still runs: decisions come from the sensors only
and there is no scene description.

## Settings (`guided_settings`)

| Key | Default | Meaning |
|-----|---------|---------|
| `autonomy` | `ask_always` | See table above |
| `confidence_threshold` | 0.7 | Used by `ask_when_unsure` |
| `answer_timeout_s` | 30 | How long to wait for your answer |
| `max_reproposals` | 2 | Alternatives offered after you say no |
| `listen` | `auto` | `auto`, `keyboard` or `microphone` |
| `listen_seconds` | 4 | Microphone recording length |
| `transcribe_model` | `whisper-1` | OpenAI speech-to-text model |
| `build_map` | true | Run SLAM + map stream underneath |
| `map_path` | `logs/guided_map_final.jpg` | Where the final map is saved |
| `forward_duration_s` | 1.2 | Seconds per approved forward move |
| `turn_duration_s` | 0.8 | Seconds per approved turn |
| `backup_duration_s` | 1.0 | Seconds per approved reverse |

The camera's "sudden visual change" panic is disabled in this mode; it fires
on every step when the robot walks and was the source of the repeated
"that changed suddenly" line.

## Code

- `planning/guided_explorer.py` — the loop (`GuidedExplorer`), autonomy policy,
  the AI decision request (`GuidedDecider`) and answer parsing (`interpret_answer`)
- `voice/listener.py` — keyboard / microphone input, `poll()` for interjections
- `tests/planning/test_guided_explorer.py`, `tests/voice/test_listener.py`
