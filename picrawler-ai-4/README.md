# PiCrawler-AI v4 - Robust Layered Architecture

**Complete redesign** using industry-standard robotics architecture principles.

## What's New in v4

### ✅ Sensor Fusion (WorldModel)
- **Before**: AI and sensors conflicted
- **After**: Sensors fused into unified world model BEFORE AI sees it
- **Result**: AI makes informed decisions with complete picture

### ✅ Spatial Memory (Learning)
- **Before**: No memory, repeated same failures
- **After**: Learns from history, avoids failed strategies
- **Result**: Gets smarter over time, natural stuck detection

### ✅ Behavior Trees (Structured Decisions)
- **Before**: Linear flow with random overrides
- **After**: Hierarchical priorities with fallbacks
- **Result**: Predictable, testable behavior

### ✅ Layered Control
- **Before**: AI tried to control motors directly
- **After**: Fast reactive → Behavior tree → AI strategy
- **Result**: Safe, responsive, intelligent

## Architecture

```
┌─────────────────────────┐
│   AI Planner (5s)       │  ← High-level strategy
│   Multi-step plans      │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Behavior Tree (500ms)  │  ← Action selection
│  Structured fallbacks   │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  WorldModel (50ms)      │  ← Sensor fusion
│  Ultrasonic + Vision    │
└────────────┬────────────┘
             ↓
         🤖 ROBOT
```

## Quick Start

```bash
# Setup (first time)
cd picrawler-ai-4
cp config/config.example.json config/config.json
nano config/config.json  # Add your OpenAI API key

# Run standard exploration
python main.py --mode explore --duration 5 --verbose

# Run cautious mode (shorter movements, more careful)
python main.py --mode cautious --duration 5

# Test (one frame + analysis)
python main.py --mode test
```

## Directory Structure

```
picrawler-ai-4/
├── core/                  # Core infrastructure
│   ├── world_model.py         # Sensor fusion
│   ├── spatial_memory.py      # History & learning
│   └── robot_controller.py    # Hardware abstraction
│
├── perception/            # Sensing & understanding
│   ├── camera.py              # Camera capture
│   └── vision_ai.py           # Scene analysis
│
├── planning/              # Decision making
│   ├── behavior_tree.py       # Behavior tree framework
│   └── ai_planner.py          # High-level strategy
│
├── config/                # Configuration
│   └── config.example.json    # Template config
│
├── logs/                  # Output
│   └── operation_v4.log       # Runtime logs
│
└── main.py                # Main control loop
```

## Key Components

### WorldModel (core/world_model.py)
**Purpose**: Unified sensor fusion

**Features**:
- Combines ultrasonic + vision into single view
- Obstacle detection in 4 directions
- Suggests best direction based on all sensors
- Calculates free space score (0=trapped, 1=open)

**Usage**:
```python
world_model = WorldModel(obstacle_threshold_cm=20)
world_model.update_ultrasonic(15.5)  # Physical sensor
world_model.update_vision(objects, hazards, description)  # Vision AI
is_safe = world_model.is_safe_to_move('forward')
```

---

### SpatialMemory (core/spatial_memory.py)
**Purpose**: Learn from experience

**Features**:
- Tracks action history and outcomes
- Learns which directions work better
- Detects stuck patterns:
  - Same action repeating
  - Oscillation (left-right-left-right)
  - No spatial progress
  - Failed escape attempts
- Suggests action scores based on history

**Usage**:
```python
memory = SpatialMemory()
memory.record_action('forward', success=False, reason="blocked")
is_stuck = memory.is_stuck()
best_turn = memory.get_best_turn_direction()
```

---

### Behavior Trees (planning/behavior_tree.py)
**Purpose**: Structured decision making

**Features**:
- SequenceNode: Do A, then B, then C (fail if any fail)
- FallbackNode: Try A, if fails try B, if fails try C
- Pre-built exploration trees
- Composable, testable behaviors

**Tree Structure**:
```
Exploration Root (Fallback)
├─ Stuck Recovery (Sequence)
│  ├─ Check if stuck
│  └─ Execute recovery
├─ Move Forward (Sequence)
│  ├─ Check path clear
│  └─ Move forward
├─ Find Alternative (Sequence)
│  ├─ Smart turn (uses memory)
│  ├─ Check path clear
│  └─ Move forward
└─ Aggressive Maneuver (Sequence)
   ├─ Back up
   ├─ Smart turn
   └─ Move forward
```

---

### AI Planner (planning/ai_planner.py)
**Purpose**: High-level strategy (not motor control)

**Input**:
- Full world model state
- Complete action history
- Memory statistics

**Output**:
```json
{
  "primary": ["turn_left", "forward", "forward"],
  "fallback": ["turn_right", "forward"],
  "recovery": ["backward", "turn_left"],
  "reasoning": "Front blocked, left has more clearance",
  "confidence": 0.85
}
```

**When Used**:
- Every 5 seconds
- After 2+ consecutive failures
- Can request replanning anytime

---

## How It Works

### Control Loop (60ms cycle)

```python
while running:
    # 1. PERCEPTION (Fast - 50ms)
    distance = robot.get_distance()        # Ultrasonic
    world_model.update_ultrasonic(distance)

    # Every 2.5s:
    image = camera.capture()               # Camera
    analysis = vision_ai.analyze(image)   # Vision AI
    world_model.update_vision(analysis)   # Fuse

    # 2. BEHAVIOR (Medium - 500ms)
    context = BehaviorContext(world_model, memory, robot)
    status = behavior_tree.execute(context)

    # 3. PLANNING (Slow - 5s, when needed)
    if should_replan():
        plan = ai_planner.plan(world_model, memory)

    # 4. LEARNING
    memory.record_outcomes()
```

### Decision Flow

```
Sensor reads 15cm obstacle
    ↓
WorldModel: "Front blocked (ultrasonic, high confidence)"
    ↓
BehaviorTree: "Path not clear, try alternative"
    ↓
SmartTurn: Memory says right worked better
    ↓
Execute: turn_right 0.9s
    ↓
Memory: Record turn_right success
```

## Differences from v3

| Aspect | v3 (Old) | v4 (New) |
|--------|----------|----------|
| **Decision** | AI → Override | Sense → Plan → Act |
| **Sensors** | Separate streams | Fused into WorldModel |
| **Actions** | Single action | Multi-step plans |
| **Fallbacks** | Random override | Behavior tree priorities |
| **Memory** | None | SpatialMemory learning |
| **Stuck** | Counter > 3 | Pattern recognition |
| **AI Role** | Motor control | High-level strategy |
| **Testability** | Hard (monolithic) | Easy (modular) |

## Testing

### Test Sensor Fusion
```python
from core.world_model import WorldModel

world = WorldModel()
world.update_ultrasonic(15.0)  # Close obstacle
world.update_vision([], ['wall'], "Wall ahead")

print(world.is_safe_to_move('forward'))  # False
print(world.get_best_direction())        # 'turn_left' or 'turn_right'
```

### Test Spatial Memory
```python
from core.spatial_memory import SpatialMemory

memory = SpatialMemory()

# Simulate stuck pattern
for _ in range(10):
    memory.record_action('forward', success=False, reason="blocked")

print(memory.is_stuck())  # True
```

### Test Behavior Tree
```python
from planning.behavior_tree import build_exploration_tree, BehaviorContext

tree = build_exploration_tree()
context = BehaviorContext(world_model, memory, robot)
status = tree.execute(context)
```

## Configuration

**Obstacle Threshold**:
```json
"obstacle_distance_threshold_cm": 20  // Stop if < 20cm
```
- Lower (10-15): More aggressive
- Higher (25-30): More cautious

**Camera Interval**:
```json
"capture_interval_s": 2.5  // Update vision every 2.5s
```
- Lower (1.0): More responsive, more API calls
- Higher (5.0): Less responsive, fewer API calls

## Expected Performance

- **Stuck incidents**: Rare (behavior tree + memory prevent loops)
- **Escape success**: ~90% (proper recovery sequences)
- **Coverage**: High (efficient exploration)
- **Decision latency**: 500ms (behavioral) + 5s (replanning)

## Migration from v3

v3 still works! v4 is a complete redesign.

**To migrate**:
1. Keep v3 as backup
2. Copy config to v4
3. Test v4 in parallel
4. Compare results

**v4 is better if you want**:
- Fewer stuck situations
- Learning from experience
- More predictable behavior
- Easier to debug/test

## Troubleshooting

**Robot gets stuck**:
- Check logs for stuck detection
- Memory should trigger recovery
- Verify sensor fusion is working

**AI not used**:
- Behavior tree handles most decisions
- AI only used for strategic replanning
- This is intentional (faster, more reliable)

**Sensor fusion not working**:
- Check `world_model.obstacles['front']`
- Should show distance and confidence
- Both ultrasonic and vision should contribute

## Future Enhancements

- [ ] SLAM / position tracking
- [ ] Multi-robot coordination
- [ ] Web dashboard for monitoring
- [ ] Replay system from logs
- [ ] More behavior trees (search, mapping, etc.)

---

## Philosophy

**v3**: "AI controls everything, override when wrong"
**v4**: "Sensors → Behavior → AI strategy, each layer does what it's good at"

The robot is now **collaborative** (layers work together) not **combative** (layers fight each other).
