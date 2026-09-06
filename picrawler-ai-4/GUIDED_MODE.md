# Guided Mode

The robot looks around, tells you what it sees, proposes what to do next and
asks you out loud before doing it.

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

You answer by typing in the terminal and pressing Enter. The robot speaks
through the speaker (OpenAI text-to-speech via `voice_settings`).

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

## Answering by voice instead of keyboard

If the robot has a USB microphone, set in `config/config.json`:

```json
"guided_settings": {
  "listen": "microphone",
  "listen_seconds": 4,
  "transcribe_model": "whisper-1"
}
```

After each question the robot records `listen_seconds` of audio with
`arecord` and transcribes it with OpenAI. If recording fails or nothing is
heard, it falls back to the keyboard automatically.

Check the microphone works first:

```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav && aplay test.wav
```

## Settings (`guided_settings`)

| Key | Default | Meaning |
|-----|---------|---------|
| `answer_timeout_s` | 30 | How long to wait for your answer |
| `max_reproposals` | 2 | Alternatives offered after you say no |
| `listen` | `keyboard` | `keyboard` or `microphone` |
| `listen_seconds` | 4 | Microphone recording length |
| `transcribe_model` | `whisper-1` | OpenAI speech-to-text model |
| `forward_duration_s` | 1.2 | Seconds per approved forward move |
| `turn_duration_s` | 0.8 | Seconds per approved turn |
| `backup_duration_s` | 1.0 | Seconds per approved reverse |

The camera's "sudden visual change" panic is disabled in this mode; it fires
on every step when the robot walks and was the source of the repeated
"that changed suddenly" line.

## Code

- `planning/guided_explorer.py` — the loop (`GuidedExplorer`), the AI decision
  request (`GuidedDecider`) and answer parsing (`interpret_answer`)
- `voice/listener.py` — keyboard / microphone input
- `tests/planning/test_guided_explorer.py`, `tests/voice/test_listener.py`
