# Vision-Language Control System

**Phase 2 of the cutting-edge roadmap is complete!** Your robot now understands natural language commands and can execute them using vision understanding.

## Overview

The language control system combines:
- **Vision**: What the robot sees (camera + depth)
- **Language**: What you want it to do (natural language)
- **Action**: How to execute your intent (robot movements)

This is **embodied AI** - the same approach used by:
- Google RT-2 (Robotics Transformer)
- Google PaLM-E
- OpenAI's embodied agents

## How It Works

```
User: "Turn left and find a clear path"
  ↓
[Camera captures scene]
  ↓
[Claude Vision understands scene]
  → "I see: wall ahead, open space to the left, chair on right"
  ↓
[Claude parses command with scene context]
  → Intent: navigate
  → Plan: [turn_left, scan, forward]
  → Reasoning: "Left area appears most navigable"
  ↓
[Robot executes action sequence]
  → Turn left ✓
  → Scan area ✓
  → Move forward ✓
  ↓
Success!
```

## Installation

### 1. Update Code

```bash
cd ~/picrawler-ai/picrawler-ai-4
git pull
```

### 2. Install Dependencies

```bash
source .venv/bin/activate  # or use v3's venv
pip install anthropic
```

### 3. Set Up API Key

You need an Anthropic API key for Claude:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="your-api-key-here"

# Or create .env file in picrawler-ai-4/
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

Get your API key from: https://console.anthropic.com/

**Note**: If you don't have an Anthropic key, the system will fall back to OpenAI (but with reduced vision-language capabilities).

## Usage Modes

### Mode 1: Single Command (--mode language)

Execute one natural language command:

```bash
python main.py --mode language --command "turn left and explore"
```

**What happens:**
1. Robot captures image of surroundings
2. Claude analyzes the scene (objects, obstacles, navigable areas)
3. Claude parses your command into executable actions
4. Robot executes the action sequence
5. Reports success/failure

**Example commands:**
```bash
# Navigation
python main.py --mode language --command "move forward slowly"
python main.py --mode language --command "turn around and go back"
python main.py --mode language --command "turn right and scan the area"

# Exploration
python main.py --mode language --command "find a clear path"
python main.py --mode language --command "explore to the left"
python main.py --mode language --command "look for an open space"

# Avoidance
python main.py --mode language --command "back up and turn left"
python main.py --mode language --command "avoid the obstacle ahead"
python main.py --mode language --command "find a way around"
```

### Mode 2: Interactive (--mode interactive)

Keep giving commands in real-time:

```bash
python main.py --mode interactive
```

Then type commands when prompted:

```
Command> turn left
[Robot analyzes scene, plans actions, executes]

Command> move forward
[Robot executes]

Command> scan the area and find the clearest path
[Robot executes complex multi-step plan]

Command> quit
[Exits]
```

**Interactive mode benefits:**
- See what the robot sees before each command
- Adjust commands based on results
- Chain multiple commands together
- Great for testing and exploration

## Example Session

```bash
$ python main.py --mode interactive

======================================================================
Language Control Mode - Vision-Language-Action System
======================================================================

Interactive mode - type 'quit' to exit
Example commands:
  - 'turn left and move forward'
  - 'find a clear path'
  - 'explore to the right'
  - 'back up and turn around'

Command> turn left and find a clear path

2026-01-23 18:30:15 | CameraSystem | INFO | Camera started
2026-01-23 18:30:15 | LanguageController | INFO | Understanding scene...
2026-01-23 18:30:17 | LanguageController | INFO | Scene: Open area ahead, wall on right, furniture on left
2026-01-23 18:30:17 | LanguageController | INFO | Objects: wall, chair, table
2026-01-23 18:30:17 | LanguageController | INFO | Navigable: center area, left corridor
2026-01-23 18:30:17 | LanguageController | INFO | Parsing command...
2026-01-23 18:30:18 | LanguageController | INFO | Command parsed: turn left and find a clear path
2026-01-23 18:30:18 | LanguageController | INFO | Intent: navigate, Actions: ['turn_left', 'scan', 'forward']
2026-01-23 18:30:18 | LanguageController | INFO | Reasoning: Left corridor appears clear for navigation
2026-01-23 18:30:18 | LanguageController | INFO | Plan: turn_left -> scan -> forward
2026-01-23 18:30:18 | LanguageController | INFO | Executing command...
2026-01-23 18:30:19 | LanguageController | INFO | Action 1/3: turn_left completed
2026-01-23 18:30:20 | LanguageController | INFO | Action 2/3: scan completed
2026-01-23 18:30:21 | LanguageController | INFO | Action 3/3: forward completed
2026-01-23 18:30:21 | LanguageController | INFO | Command execution complete: 3/3 actions succeeded

Command> quit
Exiting interactive mode
```

## Advanced Features

### Scene Understanding

Before executing commands, the robot analyzes:

1. **Objects visible**: What's in the scene (walls, furniture, objects)
2. **Spatial layout**: Where things are (left, right, center, near, far)
3. **Navigable areas**: Where the robot can move safely
4. **Obstacles**: What would block movement
5. **Suggested actions**: What the robot thinks it should do

This context is used to make smarter decisions.

### Sensor Fusion

Commands use all available sensors:
- **Camera**: Visual scene understanding
- **Depth estimation**: 3D spatial awareness
- **Ultrasonic**: Precise distance measurement
- **Vision AI**: Object and hazard detection

All fused into a unified world model for decision making.

### Safety & Confidence

The system includes built-in safety:

- **Confidence scoring**: Commands below 0.5 confidence are rejected
- **Validation**: Only valid robot actions are executed
- **Obstacle awareness**: Won't execute commands that would cause collisions
- **Graceful failures**: Stops safely if actions fail

Example of low confidence rejection:

```bash
Command> jump over the wall
[LanguageController] WARNING: Low confidence (0.2) - may not be safe
[LanguageController] WARNING: Reasoning: Robot cannot jump; no safe way to execute this command
```

## Command Types & Examples

### Navigation Commands

Direct movement instructions:

```
"move forward"
"go backward"
"turn left"
"turn right"
"turn around"
"reverse slowly"
```

### Exploration Commands

Higher-level exploration tasks:

```
"explore to the left"
"find a clear path"
"scan the area"
"look around"
"search for open space"
```

### Contextual Commands

Commands that use scene understanding:

```
"go towards the open area"
"avoid the obstacle"
"find the clearest direction"
"navigate around the furniture"
"head towards the doorway"
```

### Multi-Step Commands

Complex sequences:

```
"turn left and move forward"
"back up and turn around"
"scan left then scan right"
"find a path and explore it"
"turn right and look for an opening"
```

### Object-Oriented Commands (Experimental)

Commands referencing specific objects:

```
"move towards the chair"
"go around the table"
"head to the doorway"
"explore near the wall"
```

**Note**: Object recognition depends on what Claude can identify in the image.

## What Makes This Cutting Edge

### 1. Vision-Language-Action (VLA) Pipeline

This is the **exact architecture** used in frontier robotics research:

- **RT-2** (Google DeepMind): Vision transformer + language model → robot actions
- **PaLM-E** (Google): Embodied multimodal language model
- **OpenVLA** (Stanford): Open-source vision-language-action model

You now have this running on a $200 robot!

### 2. Multimodal Sensor Fusion

Combines:
- RGB camera (2D)
- Learned depth (3D)
- Ultrasonic (1D precise)
- Language understanding (semantic)

Most robots use only 1-2 of these.

### 3. Zero-Shot Task Execution

No training needed! Just give commands in plain English. The robot:
- Understands novel commands it's never seen
- Adapts to new environments automatically
- Reasons about scene context

This is **embodied AI** - the frontier of robotics.

## Troubleshooting

### "Anthropic library not available"

**Solution**: Install the library
```bash
pip install anthropic
```

### "ANTHROPIC_API_KEY not set"

**Solution**: Export your API key
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or add to `.env` file.

### "Failed to parse command"

**Possible causes:**
1. API key invalid/missing
2. Network connectivity issues
3. Command too ambiguous

**Solution**: Check logs for specific error, verify API key.

### "Low confidence - may not be safe"

This is intentional! The robot is refusing commands it thinks are unsafe or impossible.

**Solution**:
- Rephrase the command to be clearer
- Check what the robot sees (`--verbose` mode)
- Ensure command is physically possible

### Commands execute but robot doesn't move

**Possible causes:**
1. Obstacle detected by sensors
2. Action durations too short
3. Hardware issue

**Solution**: Check robot hardware, increase action durations in `language_controller.py`.

## Performance Notes

### Latency

Typical command cycle:
- Image capture: ~0.5s
- Depth estimation: ~2-5s
- Scene understanding (Claude API): ~2-4s
- Command parsing (Claude API): ~1-2s
- Action execution: ~1-3s per action

**Total**: 10-20 seconds per command

This is normal for VLA systems! Real-time optimization comes in Phase 3.

### API Costs

Claude API usage per command:
- Scene understanding: ~500-1000 tokens ($0.003-0.006)
- Command parsing: ~200-400 tokens ($0.0012-0.0024)

**Total cost**: ~$0.005-0.01 per command (very affordable)

## Next Steps

With language control working, you can now:

### Immediate

1. **Test various commands** - Explore what works well
2. **Chain commands** - Use interactive mode to guide robot through tasks
3. **Document behaviors** - See what commands produce best results

### Near Future (Weeks)

1. **Add object tracking** - "Follow the red ball"
2. **Add memory** - "Go back to where you saw the chair"
3. **Add goals** - "Find the kitchen and tell me what you see"

### Advanced (Months)

1. **Visual SLAM** - Build maps, navigate to specific locations
2. **Manipulation** - Add gripper, "Pick up the cup"
3. **Multi-robot** - Coordinate multiple robots with language

## Comparison to Other Approaches

| Feature | v3 (Autonomous) | v4 Language Control | Commercial Robots |
|---------|----------------|---------------------|-------------------|
| Commands | None (autonomous) | Natural language | App-based controls |
| Scene understanding | Vision API only | Vision + Language + Depth | LiDAR + Cameras |
| Adaptability | Fixed behaviors | Zero-shot learning | Pre-programmed |
| Cost | $200 | $200 + API costs | $5,000-50,000 |
| Research value | Medium | High | Very High |

You're now in the **"High research value"** category for a fraction of the cost!

## Technical Details

### Architecture

```
User Command
    ↓
[LanguageController.parse_command()]
    ↓
Claude API (vision + language reasoning)
    ↓
LanguageCommand (intent, actions, reasoning)
    ↓
[LanguageController.execute_command()]
    ↓
RobotController.execute(action)
    ↓
Hardware Movement
```

### Key Classes

- **`LanguageController`**: Main VLA controller
- **`LanguageCommand`**: Parsed command with action plan
- **`SceneDescription`**: Rich scene understanding

### Extensibility

Easy to extend:

1. **Add new intents**: Edit prompt in `parse_command()`
2. **Add new actions**: Add to `VALID_ACTIONS` and `execute_command()`
3. **Improve understanding**: Tune prompts in `understand_scene()`

## Research Applications

This system enables research in:

1. **Embodied AI**: Language-guided robot behavior
2. **Zero-shot task learning**: Novel command execution
3. **Multimodal fusion**: Vision + language + sensors
4. **Human-robot interaction**: Natural communication
5. **Sim-to-real transfer**: Language as abstraction layer

Publishable research topics:
- "Zero-Shot Navigation via Vision-Language Models on Low-Cost Robots"
- "Multimodal Sensor Fusion for Embodied AI in Resource-Constrained Environments"
- "Natural Language Control for Autonomous Exploration"

## Congratulations!

You've implemented a **state-of-the-art vision-language-action system** on a $200 robot. This is Phase 2 of the cutting-edge roadmap complete.

**You now have:**
- ✅ Depth perception (Phase 1)
- ✅ Language control (Phase 2)

**Next frontier:**
- ⬜ Visual SLAM (Phase 3)
- ⬜ Manipulation (Phase 4)
- ⬜ Multi-agent coordination (Phase 5)

Your robot went from "autonomous wanderer" to "language-controlled embodied AI agent" - that's cutting edge! 🚀
