# PiCrawler-AI: Robust Architecture Redesign

## Executive Summary

The current architecture is **reactive and single-threaded** with AI making isolated decisions without full context. A robust redesign would use a **layered hybrid architecture** with sensor fusion, behavior trees, spatial memory, and proper separation of concerns.

---

## Current Architecture Issues

### Problem 1: Inverted Control Flow
```
Current (Broken):
1. AI decides "forward" (doesn't know about obstacle)
2. Robot checks sensor (oh no, obstacle!)
3. Override decision (random turn)
4. Repeat infinitely

Result: AI and sensor fighting each other
```

### Problem 2: No State Awareness
- Robot has no memory of where it's been
- Can't distinguish "I'm in a corner" from "I'm in open space"
- No concept of "tried this 5 times, time for new strategy"

### Problem 3: Single Action Decisions
- AI commits to ONE action per cycle
- No backup plans, no sequences
- Forces random overrides when blocked

### Problem 4: Late Sensor Integration
- Sensor data passed to AI as JSON afterthought
- AI trained on text, not good at spatial reasoning
- Camera and ultrasonic are separate, not fused

---

## Redesigned Architecture

### Layered Hybrid Architecture (Industry Standard)

```
┌─────────────────────────────────────────────────┐
│         HIGH-LEVEL PLANNER (AI)                 │
│  "Explore room" → ["move_to_open_space",        │
│                    "scan_area", "navigate"]     │
└────────────────┬────────────────────────────────┘
                 │ Goal/Task
                 ▼
┌─────────────────────────────────────────────────┐
│       BEHAVIOR COORDINATOR                      │
│  Behavior Trees + State Machine                 │
│  - Exploration behavior                         │
│  - Obstacle avoidance behavior                  │
│  - Recovery behavior                            │
│  - Stuck detection behavior                     │
└────────────────┬────────────────────────────────┘
                 │ Primitive Actions
                 ▼
┌─────────────────────────────────────────────────┐
│         SENSOR FUSION LAYER                     │
│  Combines: Camera + Ultrasonic + Odometry       │
│  Output: Unified world model                    │
│  - Obstacle map (360°)                          │
│  - Free space estimation                        │
│  - Confidence levels                            │
└────────────────┬────────────────────────────────┘
                 │ World State
                 ▼
┌─────────────────────────────────────────────────┐
│      REACTIVE CONTROL LAYER                     │
│  Fast, deterministic safety behaviors           │
│  - Emergency stop on collision                  │
│  - Cliff detection                              │
│  - Hardware limits                              │
└────────────────┬────────────────────────────────┘
                 │ Motor Commands
                 ▼
         [ Physical Robot ]
```

---

## Component Redesign

### 1. Sensor Fusion Module (New)

**Purpose**: Create single unified world model BEFORE AI sees it

```python
class WorldModel:
    """Unified representation of robot's environment"""

    def __init__(self):
        self.obstacle_distances = {
            'front': None,      # From ultrasonic
            'left': None,       # From turning + sensing
            'right': None,      # From turning + sensing
            'rear': None        # Estimated from movement
        }
        self.visual_features = []   # From camera AI
        self.free_space_score = 0.0  # 0=trapped, 1=open
        self.confidence = 0.0        # Data quality

    def is_safe_to_move(self, direction: str) -> bool:
        """Multi-sensor decision, not just one sensor"""
        if direction == 'forward':
            # Check ultrasonic first (hard constraint)
            if self.obstacle_distances['front'] < 20:
                return False
            # Check vision AI (soft constraint)
            if 'wall' in self.visual_features:
                return False
            return True
        # Similar logic for other directions

    def suggest_best_direction(self) -> str:
        """Based on ALL sensors, which way is most open?"""
        scores = {}
        for direction, distance in self.obstacle_distances.items():
            if distance is None:
                scores[direction] = 0.5  # Unknown
            else:
                scores[direction] = min(distance / 100.0, 1.0)
        return max(scores, key=scores.get)
```

**Benefits**:
- AI sees unified view, not conflicting data
- Sensor readings trump AI guesses (correct priority)
- Easy to add more sensors later (lidar, cliff sensors, etc.)

---

### 2. Behavior Tree Coordinator (New)

**Purpose**: Replace linear decision flow with hierarchical behaviors

```python
class BehaviorNode:
    """Base class for behavior tree nodes"""
    def execute(self, world_model: WorldModel) -> Status:
        raise NotImplementedError

class SequenceNode(BehaviorNode):
    """Execute children in order until one fails"""
    def __init__(self, children: List[BehaviorNode]):
        self.children = children

    def execute(self, world_model):
        for child in self.children:
            status = child.execute(world_model)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

class FallbackNode(BehaviorNode):
    """Try children until one succeeds (priority list)"""
    def __init__(self, children: List[BehaviorNode]):
        self.children = children

    def execute(self, world_model):
        for child in self.children:
            status = child.execute(world_model)
            if status != Status.FAILURE:
                return status
        return Status.FAILURE

# Build behavior tree:
exploration_tree = FallbackNode([
    # Priority 1: Safety first
    SequenceNode([
        CheckStuckCondition(),
        ExecuteRecoverySequence()
    ]),
    # Priority 2: Try to move forward
    SequenceNode([
        CheckPathClear(),
        MoveForward()
    ]),
    # Priority 3: Find alternative path
    SequenceNode([
        FindBestTurnDirection(),  # Uses world model
        ExecuteTurn(),
        MoveForward()
    ]),
    # Priority 4: Aggressive recovery
    BackUpAndReorient()
])
```

**Benefits**:
- Clear behavior priorities (not random)
- Built-in fallback strategies
- Composable and testable
- Industry-proven pattern

---

### 3. Spatial Memory System (New)

**Purpose**: Remember where you've been, avoid repeating failures

```python
class SpatialMemory:
    """Track robot's history and learn from failures"""

    def __init__(self):
        # Simple occupancy grid (or pose history)
        self.visited_positions = deque(maxlen=50)
        self.failed_directions = defaultdict(int)  # direction → fail count
        self.action_history = deque(maxlen=20)
        self.last_escape = None

    def record_action(self, action: str, world_state: WorldModel,
                     success: bool):
        """Learn from outcomes"""
        self.action_history.append({
            'action': action,
            'state': world_state.to_dict(),
            'success': success,
            'timestamp': time.time()
        })

        if not success:
            self.failed_directions[action] += 1

    def get_action_recommendation(self) -> Dict[str, float]:
        """Return action scores based on history"""
        scores = {
            'forward': 1.0,
            'turn_left': 1.0,
            'turn_right': 1.0,
            'backward': 0.5  # Less preferred but valid
        }

        # Penalize recently failed actions
        for action, fail_count in self.failed_directions.items():
            decay = 0.9 ** fail_count  # Exponential decay
            scores[action] *= decay

        # Penalize repeating same action
        recent_actions = [a['action'] for a in self.action_history[-5:]]
        if len(set(recent_actions)) == 1:  # All same action
            repeated_action = recent_actions[0]
            scores[repeated_action] *= 0.2  # Heavy penalty

        return scores

    def is_stuck(self) -> bool:
        """Sophisticated stuck detection"""
        if len(self.action_history) < 10:
            return False

        recent = self.action_history[-10:]

        # Check 1: Repeating same action with failures
        actions = [a['action'] for a in recent]
        if actions.count(actions[0]) >= 7:  # Same action 7/10 times
            return True

        # Check 2: Oscillating (left, right, left, right...)
        if self._is_oscillating(actions):
            return True

        # Check 3: Spatial stagnation (not moving in world)
        if self._not_making_progress():
            return True

        return False

    def clear_failed_directions(self):
        """Reset after successful escape"""
        self.failed_directions.clear()
```

**Benefits**:
- Robot learns from experience
- Avoids repeating failed strategies
- Sophisticated stuck detection
- Helps AI make better decisions

---

### 4. AI Planner (Redesigned)

**Purpose**: High-level goal planning, not low-level motor control

```python
class AIPlanner:
    """AI for high-level strategy, not motor control"""

    def plan_exploration_strategy(self,
                                  world_model: WorldModel,
                                  spatial_memory: SpatialMemory) -> Plan:
        """Generate multi-step plan"""

        # Build rich context for AI
        context = {
            'current_state': {
                'obstacles': world_model.obstacle_distances,
                'visual': world_model.visual_features,
                'free_space': world_model.free_space_score
            },
            'history': {
                'recent_actions': spatial_memory.action_history[-10:],
                'failed_directions': dict(spatial_memory.failed_directions),
                'stuck_status': spatial_memory.is_stuck()
            },
            'action_preferences': spatial_memory.get_action_recommendation()
        }

        prompt = f"""
        You are a robot navigation planner. Based on the current state,
        create a multi-step plan to explore effectively.

        Current situation:
        - Front obstacle: {context['current_state']['obstacles']['front']}cm
        - Recent actions: {[a['action'] for a in context['history']['recent_actions']]}
        - Failed directions: {context['history']['failed_directions']}
        - Stuck: {context['history']['stuck_status']}

        Return a plan with:
        1. Primary action sequence (3-5 steps)
        2. Fallback if primary fails
        3. Recovery strategy if both fail

        Format:
        {{
            "primary": ["action1", "action2", "action3"],
            "fallback": ["alt_action1", "alt_action2"],
            "recovery": ["recovery_action"],
            "reasoning": "why this plan makes sense"
        }}
        """

        # AI returns PLAN, not single action
        plan = self.client.generate(prompt)
        return Plan(plan)

    def should_replan(self, execution_state: ExecutionState) -> bool:
        """Decide if current plan is still valid"""
        if execution_state.consecutive_failures >= 2:
            return True
        if execution_state.world_changed_significantly():
            return True
        return False
```

**Benefits**:
- AI does what it's good at: high-level reasoning
- Hardware does what it's good at: fast reactive control
- Multi-step planning reduces thrashing
- Clear separation of concerns

---

### 5. Execution Controller (New)

**Purpose**: Execute plans with validation and recovery

```python
class ExecutionController:
    """Executes plans with validation and recovery"""

    def __init__(self, robot: RobotController,
                 world_model: WorldModel,
                 spatial_memory: SpatialMemory):
        self.robot = robot
        self.world_model = world_model
        self.memory = spatial_memory
        self.current_plan = None
        self.plan_index = 0

    def execute_plan(self, plan: Plan) -> bool:
        """Execute multi-step plan"""
        self.current_plan = plan
        self.plan_index = 0

        # Try primary sequence
        if self._execute_sequence(plan.primary):
            return True

        # Primary failed, try fallback
        logger.warning("Primary plan failed, trying fallback")
        if self._execute_sequence(plan.fallback):
            return True

        # Both failed, execute recovery
        logger.error("Both plans failed, executing recovery")
        return self._execute_sequence(plan.recovery)

    def _execute_sequence(self, actions: List[str]) -> bool:
        """Execute action sequence with validation"""
        for action in actions:
            # PRE-VALIDATION: Check if action is safe BEFORE trying
            if not self._validate_action(action):
                logger.warning(f"Action {action} failed pre-validation")
                self.memory.record_action(action, self.world_model, False)
                return False

            # Execute action
            logger.info(f"Executing: {action}")
            self.robot.execute(action, duration=self._get_duration(action))

            # Update world model (sense after acting)
            self.world_model.update()

            # POST-VALIDATION: Did it work?
            if self._action_succeeded(action):
                self.memory.record_action(action, self.world_model, True)
            else:
                logger.warning(f"Action {action} failed post-validation")
                self.memory.record_action(action, self.world_model, False)
                return False

        return True

    def _validate_action(self, action: str) -> bool:
        """Check if action is safe to execute"""
        if action == 'forward':
            if not self.world_model.is_safe_to_move('forward'):
                return False
            # Check if we've failed this recently
            if self.memory.failed_directions['forward'] >= 3:
                return False
        return True

    def _action_succeeded(self, action: str) -> bool:
        """Verify action had desired effect"""
        # Could use odometry, sensor changes, etc.
        # For now, simple heuristic
        return True  # Improve with actual feedback
```

**Benefits**:
- Validates BEFORE executing (not after)
- Multi-step sequences, not single actions
- Built-in fallback logic
- Learning from outcomes

---

## Key Design Principles

### 1. Sense → Plan → Act (Not Plan → Sense → Override)

**Current (Broken)**:
```python
decision = ai.decide()        # Plan without sensing
if sensor_blocked():          # Sense too late
    override_decision()       # Override = wasted work
```

**Redesigned (Correct)**:
```python
world_model.update()          # Sense first
plan = ai.plan(world_model)   # Plan with full info
executor.execute(plan)        # Act with validation
```

### 2. Layered Intelligence (Fast → Slow)

```
Layer 1 (1ms):  Hardware safety (e-stop, limits)
Layer 2 (50ms): Reactive obstacle avoidance (sensor-based)
Layer 3 (500ms): Behavior tree execution
Layer 4 (2-5s): AI planning (only when needed)
```

Fast layers can interrupt slow layers (safety first).

### 3. Separation of Concerns

| Component | Responsibility | Speed |
|-----------|---------------|-------|
| AI Planner | High-level goals | Slow (5s) |
| Behavior Tree | Action selection | Medium (500ms) |
| Sensor Fusion | World model | Fast (50ms) |
| Motor Control | Hardware commands | Very fast (1ms) |

Each component has ONE job, easy to test.

### 4. Memory and Learning

Robot remembers:
- Where it's been (spatial memory)
- What failed (action history)
- How to escape (recovery sequences)
- Progress toward goal

This prevents infinite loops naturally.

---

## Implementation Roadmap

### Phase 1: Sensor Fusion (Week 1)
- Create `WorldModel` class
- Combine ultrasonic + vision into unified representation
- Validate with sensor showing on screen

**Outcome**: Robot has single source of truth about environment

### Phase 2: Spatial Memory (Week 2)
- Implement `SpatialMemory` class
- Track action history and outcomes
- Add stuck detection based on history

**Outcome**: Robot learns from repeated failures

### Phase 3: Behavior Trees (Week 3)
- Implement basic behavior tree nodes
- Create exploration behavior tree
- Replace linear decision flow

**Outcome**: Structured fallback behaviors, no random overrides

### Phase 4: Execution Controller (Week 4)
- Implement plan execution with validation
- Add pre-flight checks before actions
- Implement multi-step sequences

**Outcome**: Actions validated before execution

### Phase 5: AI Planner Redesign (Week 5)
- Modify AI to return plans, not actions
- Add rich context (history + memory)
- Implement replanning logic

**Outcome**: AI does high-level strategy, not motor control

---

## Quick Wins (Can Do Today)

### A. Sensor Fusion Lite
```python
def get_unified_obstacle_status() -> dict:
    """Combine all obstacle info before AI sees it"""
    ultrasonic = robot.get_distance()
    vision_hazards = analysis.hazards

    # Create unified view
    return {
        'forward_blocked': ultrasonic < 20 or 'wall' in vision_hazards,
        'confidence': 'high' if ultrasonic else 'medium',
        'primary_reason': 'sensor' if ultrasonic < 20 else 'vision'
    }
```

### B. Simple Action History
```python
# In base_behavior.py
self.action_outcomes = []

def record_outcome(action, blocked):
    self.action_outcomes.append({
        'action': action,
        'blocked': blocked,
        'time': time.time()
    })

    # Pass to AI
    recent_failures = [
        a for a in self.action_outcomes[-10:]
        if a['blocked']
    ]
```

### C. Smarter Turn Selection
```python
# Instead of random turn
turn_scores = {
    'turn_left': 1.0 - (left_failures / 10),
    'turn_right': 1.0 - (right_failures / 10)
}
best_turn = max(turn_scores, key=turn_scores.get)
```

---

## Comparison: Current vs Redesigned

| Aspect | Current | Redesigned |
|--------|---------|------------|
| **Decision Flow** | AI → Override → Execute | Sense → Plan → Validate → Execute |
| **Sensor Integration** | Late (passed as JSON) | Early (fused world model) |
| **Action Planning** | Single action | Multi-step sequences |
| **Fallback Strategy** | Random override | Hierarchical behavior tree |
| **Memory** | None (memoryless) | Spatial + action history |
| **Stuck Detection** | Counter-based | Pattern recognition |
| **AI Role** | Low-level control | High-level planning |
| **Testability** | Hard (monolithic) | Easy (modular) |

---

## Why This Works Better

### Problem: AI Keeps Suggesting Forward
**Current**: Override randomly to turn
**Redesigned**: Sensor fusion prevents AI from even considering forward

### Problem: Random Turns Don't Help
**Current**: `random.choice(['left', 'right'])`
**Redesigned**: Memory scores show which direction historically more successful

### Problem: No Concept of "Stuck"
**Current**: Count consecutive overrides (brittle)
**Redesigned**: Pattern recognition across multiple indicators

### Problem: Single Action Myopia
**Current**: One action per cycle, no planning ahead
**Redesigned**: Multi-step sequences with fallbacks

---

## Code Structure

```
picrawler-ai-4/  # New architecture
├── core/
│   ├── world_model.py        # NEW: Sensor fusion
│   ├── spatial_memory.py     # NEW: History tracking
│   ├── robot_controller.py   # KEEP: Hardware abstraction
│   └── execution_controller.py  # NEW: Plan execution
│
├── planning/
│   ├── ai_planner.py         # REDESIGNED: High-level goals
│   ├── behavior_tree.py      # NEW: Behavior nodes
│   └── recovery_behaviors.py # NEW: Stuck recovery
│
├── perception/
│   ├── camera.py             # KEEP: Camera capture
│   ├── sensor_fusion.py      # NEW: Combine sensors
│   └── feature_extractor.py  # NEW: Visual features
│
├── behaviors/                # REDESIGNED
│   ├── exploration.py        # Uses behavior trees now
│   ├── object_search.py
│   └── recovery.py           # NEW: Stuck recovery
│
└── main.py                   # REDESIGNED: Proper control loop
```

---

## Testing Strategy

### Unit Tests (Each Component)
```python
def test_world_model_fusion():
    model = WorldModel()
    model.update_ultrasonic(15.0)  # Obstacle
    model.update_vision(['wall'])  # Vision confirms
    assert not model.is_safe_to_move('forward')

def test_spatial_memory_stuck_detection():
    memory = SpatialMemory()
    # Simulate stuck scenario
    for _ in range(10):
        memory.record_action('forward', world_state, success=False)
    assert memory.is_stuck()
```

### Integration Tests (Components Together)
```python
def test_sensor_fusion_prevents_collision():
    world_model = WorldModel()
    world_model.update_ultrasonic(10.0)  # Very close!

    planner = AIPlanner()
    plan = planner.plan_exploration_strategy(world_model, memory)

    # Verify plan doesn't include forward
    assert 'forward' not in plan.primary
```

### Hardware-in-Loop Tests
```python
def test_real_obstacle_avoidance():
    # Place robot facing wall at 15cm
    robot = RobotController()
    world_model = WorldModel()
    executor = ExecutionController(robot, world_model, memory)

    plan = Plan(primary=['forward'])
    result = executor.execute_plan(plan)

    # Should fail pre-validation, not crash into wall
    assert not result
    assert robot.get_distance() > 10  # Didn't hit wall
```

---

## Migration Path (From Current to Redesigned)

### Option A: Gradual Migration
1. Add `WorldModel` class (runs alongside current)
2. Replace override logic with world model checks
3. Add `SpatialMemory` tracking
4. Refactor behaviors to use behavior trees
5. Finally, redesign AI integration

### Option B: Fresh Start (Recommended)
1. Keep current code as `picrawler-ai-3/`
2. Create `picrawler-ai-4/` with new architecture
3. Port hardware abstractions (keep what works)
4. Implement new design from scratch
5. Compare performance side-by-side

**Why fresh start?**
- Clean architecture is easier than refactoring spaghetti
- Can run both versions for comparison
- Less risk of breaking existing code
- Clearer learning experience

---

## Expected Improvements

| Metric | Current | Redesigned (Expected) |
|--------|---------|----------------------|
| Stuck incidents | Frequent | Rare |
| Escape success rate | ~30% | ~90% |
| Decision latency | 3-5s | 500ms (behavioral) + 5s (replanning) |
| Coverage (area explored) | Low (gets stuck) | High (efficient navigation) |
| Code maintainability | Hard | Easy (modular) |

---

## References & Further Reading

- **Behavior Trees**: [Behavior Trees in Robotics](https://arxiv.org/abs/1709.00084)
- **Hybrid Architecture**: Brooks, "A Robust Layered Control System"
- **Sensor Fusion**: Probabilistic Robotics (Thrun et al.)
- **ROS Navigation Stack**: Similar layered approach

---

## Summary: Key Takeaways

1. **Sense BEFORE planning**, not after
2. **Fuse sensors** into unified world model
3. **AI for strategy**, hardware for tactics
4. **Memory prevents loops** naturally
5. **Behavior trees** for structured fallbacks
6. **Validate before executing**, not after
7. **Modular design** = testable + maintainable

The current architecture fights itself (AI vs sensors). The redesigned architecture cooperates (AI uses sensors to plan, not override after).
