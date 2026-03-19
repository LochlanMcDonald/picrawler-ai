# Test Coverage Analysis: picrawler-ai

## Executive Summary

The codebase has **critically low test coverage**. Neither v3 nor v4 uses a test framework (pytest, unittest). All existing test files are manual integration scripts that require physical hardware to run. This means:

- No automated tests run on code changes
- No CI/CD pipeline exists
- No coverage metrics are tracked
- Large portions of business logic have zero test coverage

---

## Current State

### Existing "Tests"

| File | Type | Framework | What It Tests |
|------|------|-----------|---------------|
| `picrawler-ai-3/utils/test_ai.py` | Manual script | None | OpenAI API connectivity |
| `picrawler-ai-3/utils/test_camera.py` | Manual script | None | Camera hardware + base64 encoding |
| `picrawler-ai-3/utils/test_motors.py` | Manual script | None | Motor motion primitives |
| `picrawler-ai-3/utils/test_ultrasonic.py` | Manual script | None | Ultrasonic sensor readings |
| `picrawler-ai-4/utils/test_architecture.py` | Manual script | None | WorldModel, SpatialMemory, BehaviorTree (with mocks) |

### Code-to-Test Ratio

| Version | Source Lines | Test Lines | Ratio |
|---------|-------------|------------|-------|
| v3 | ~2,347 | ~213 | 11:1 |
| v4 | ~5,223 | ~188 | 28:1 |

---

## Proposed Improvements (Prioritised)

### Priority 1 — Critical: Testing Infrastructure

Before writing any new tests, the project needs a testing foundation.

**What to add:**
- `pytest` and `pytest-mock` as dev dependencies
- A `conftest.py` with shared fixtures (mock robot, mock OpenAI client, mock camera)
- A `pyproject.toml` or `pytest.ini` for test discovery and coverage config
- `.coveragerc` to exclude hardware-only paths from coverage reports

**Why it matters:** Without a framework, tests cannot run automatically on pull requests, and coverage cannot be measured. Every test improvement below depends on this foundation.

**Suggested file layout:**
```
picrawler-ai-3/tests/
    conftest.py
    behaviors/
        test_base_behavior.py
        test_exploration.py
        test_avoidance.py
    ai/
        test_vision_ai.py
    core/
        test_config_loader.py

picrawler-ai-4/tests/
    conftest.py
    core/
        test_world_model.py
        test_spatial_memory.py
    planning/
        test_behavior_tree.py
        test_language_controller.py
    mapping/
        test_occupancy_grid.py
        test_path_planner.py
```

---

### Priority 2 — High: v3 `BaseBehavior` Logic (562 lines, 0% coverage)

`picrawler-ai-3/behaviors/base_behavior.py` is the largest untested file and contains the most complex business logic:

- **Anti-loop detection** — tracks recent actions and penalises repeated moves
- **Stuck detection** — detects when the robot hasn't moved despite commands
- **Panic mode** — resets robot state after too many failures
- **Throttled execution** — rate-limits vision API calls
- **State machine transitions** — between explore, react, stuck, panic states

All of this logic is purely computational with no hardware dependency, making it ideal for unit testing with mocks.

**Specific test cases to write:**

```python
# Anti-loop detection
def test_repeated_actions_are_penalised():
    ...  # Same action 3x in a row → score penalty applied

def test_loop_penalty_resets_after_different_action():
    ...

# Stuck detection
def test_stuck_declared_after_n_identical_positions():
    ...

def test_stuck_clears_after_successful_movement():
    ...

# State transitions
def test_transitions_to_panic_after_max_failures():
    ...

def test_panic_resets_failure_count():
    ...
```

---

### Priority 3 — High: v4 `WorldModel` Sensor Fusion (270 lines, partially tested)

`picrawler-ai-4/core/world_model.py` fuses ultrasonic distance, vision analysis, and depth estimation into a unified robot state. The existing `test_architecture.py` touches this lightly, but key fusion logic is untested:

- **Confidence weighting** — how sensor reliability scores are combined
- **Conflict resolution** — ultrasonic says "clear", vision says "obstacle"
- **Stale data handling** — what happens when a sensor hasn't updated in N cycles
- **Threshold edge cases** — exactly at the obstacle distance boundary

**Specific test cases to write:**

```python
def test_obstacle_detected_when_ultrasonic_below_threshold():
    ...

def test_vision_overrides_ultrasonic_when_confidence_high():
    ...

def test_stale_sensor_data_reduces_confidence():
    ...

def test_fusion_returns_safe_when_all_sensors_clear():
    ...
```

---

### Priority 4 — High: v4 `OccupancyGrid` and `PathPlanner` (A\* pathfinding)

`picrawler-ai-4/mapping/occupancy_grid.py` and `path_planner.py` implement the spatial map and A\* navigation. These are **pure algorithmic code** with no hardware dependency — they are the easiest to test and highest-value.

- Occupancy grid: cell updates, obstacle inflation, boundary conditions
- Path planner: shortest path found, unreachable goal handled, path around obstacles

**Specific test cases to write:**

```python
# OccupancyGrid
def test_cell_marked_occupied_after_obstacle_update():
    ...

def test_inflation_radius_blocks_adjacent_cells():
    ...

def test_out_of_bounds_coordinates_handled_gracefully():
    ...

# PathPlanner
def test_direct_path_found_in_empty_grid():
    ...

def test_path_routes_around_obstacle():
    ...

def test_returns_none_when_goal_unreachable():
    ...

def test_start_equals_goal_returns_empty_path():
    ...
```

---

### Priority 5 — Medium: v4 `SpatialMemory` Learning (281 lines, partially tested)

`picrawler-ai-4/core/spatial_memory.py` tracks action history to detect ineffective behaviours and learn over time. The existing test covers the happy path but misses:

- **Cooldown expiry** — actions on cooldown become available again after timeout
- **Memory pruning** — oldest entries are removed when capacity is exceeded
- **Score decay** — action scores change correctly with new observations
- **Concurrent access** — thread-safety of reads/writes (used from multiple threads in main)

---

### Priority 6 — Medium: v4 `BehaviorTree` Node Types

`picrawler-ai-4/planning/behavior_tree.py` implements a behaviour tree with Selector, Sequence, and Action nodes. The existing test only validates top-level execution flow. Missing tests:

- **Selector short-circuits on first success** — remaining children not evaluated
- **Sequence aborts on first failure** — remaining children not evaluated
- **Decorator nodes** (if any) — inversion, repeat-until
- **Blackboard data sharing** — nodes reading/writing shared state correctly

---

### Priority 7 — Medium: v4 `LanguageController` Command Parsing

`picrawler-ai-4/planning/language_controller.py` translates natural language commands into robot actions. This involves prompt construction and response parsing — both fully testable without real API calls using `unittest.mock`.

**Specific test cases to write:**

```python
def test_forward_command_maps_to_move_forward():
    ...  # Mock OpenAI response → assert correct action selected

def test_ambiguous_command_asks_for_clarification():
    ...

def test_stop_command_always_takes_highest_priority():
    ...

def test_invalid_json_response_handled_gracefully():
    ...  # Malformed LLM output → fallback behaviour, not crash
```

---

### Priority 8 — Medium: v4 `VisualOdometry` Pose Tracking (269 lines, 0% coverage)

`picrawler-ai-4/perception/visual_odometry.py` tracks camera pose from successive frames using feature matching. It can be tested with synthetic image pairs (solid colours, simple patterns) without a real camera:

- Feature detection in known images
- Pose consistency when given identical frames (no movement)
- Graceful handling of images with no detectable features

---

### Priority 9 — Low: v3 Individual Behavior Subclasses

`picrawler-ai-3/behaviors/exploration.py`, `following.py`, `avoidance.py`, `object_detection.py` are thin (14–38 lines) and mostly delegate to `BaseBehavior`. Once `BaseBehavior` is well tested (Priority 2), these only need a few smoke tests to confirm:

- Each subclass instantiates without error
- The `run()` method calls the correct base class methods
- Mode-specific configuration (e.g. avoidance distance threshold) is applied

---

### Priority 10 — Low: Configuration Loading

`core/config_loader.py` (both versions) loads JSON config and falls back to defaults. Tests here are simple but prevent silent misconfiguration bugs:

- Valid config loads correctly
- Missing optional keys use defaults
- Missing required keys raise a clear error (not a `KeyError` buried in a stack trace)
- Malformed JSON raises a clear error

---

## What NOT to Unit Test

These components require real hardware or external services and should remain as manual/integration tests:

- Camera frame capture (requires Picamera2 or physical webcam)
- Ultrasonic sensor readings (requires GPIO hardware)
- Motor control (requires PiCrawler hardware)
- OpenAI API responses (should be mocked in unit tests; real calls are expensive)
- PyTorch MiDaS model loading in `depth_estimator.py` (large model download; test the interface, not the model)

---

## Suggested Implementation Order

1. Add `pytest`, `pytest-mock`, `pytest-cov` to both `requirements.txt` files
2. Create `conftest.py` files with `MockRobot` and `MockOpenAIClient` fixtures
3. Write tests for `OccupancyGrid` and `PathPlanner` (pure algorithms, quickest wins)
4. Write tests for `BaseBehavior` state machine (highest-value business logic)
5. Write tests for `WorldModel` sensor fusion edge cases
6. Write tests for `LanguageController` with mocked LLM responses
7. Set up a GitHub Actions workflow running `pytest --cov` on every push
