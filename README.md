# Valoscribe

**Automated Valorant VOD analysis tool for extracting structured game events and player states from match videos.**

Valoscribe uses computer vision and OCR to convert Valorant esports VODs into structured data, enabling analytics, highlight generation, and machine learning applications.

## Overview

Valoscribe processes YouTube VODs of professional Valorant matches to extract:

- **Game Events** - Match start/end, round start/end, kills, deaths, ability usage, ultimate casts, spike plants
- **Player States** - Frame-by-frame tracking of health, armor, shields, abilities, ultimate charge for all 10 players
- **Match Metadata** - Team names, player names, agents, starting sides (scraped from VLR.gg)

The system leverages the spectator HUD used in professional broadcasts, which displays comprehensive real-time information for all players simultaneously.

## Demo

Watch a 1-minute demo showing real-time event detection on a live round:

[![Valoscribe Demo](https://img.youtube.com/vi/xKsUO7c6Hbw/maxresdefault.jpg)](https://youtu.be/xKsUO7c6Hbw)

**[▶️ Watch Demo on YouTube](https://youtu.be/xKsUO7c6Hbw)**

The demo shows the terminal output side-by-side with the game footage, displaying events as they're detected (kills, abilities, round transitions, player states).

## Key Features

- **Fully Automated** - Provide a VLR.gg match URL, get structured event logs and frame states
- **VLR.gg Integration** - Automatic scraping of match metadata (teams, players, agents, maps)
- **YouTube Support** - Direct VOD download with timestamp support for individual maps
- **Phase Detection** - Intelligent state machine for detecting preround, active round, and post-round phases
- **Template Matching** - Robust agent detection with attack/defense variants for handling mirror compositions
- **State Validation** - Prevents false positives through multi-condition validation
- **Parallel Processing** - Process multiple matches concurrently for faster batch analysis
- **Event Finalization** - Infers missing events when VODs end early (incomplete rounds)

## Use Cases

- **Esports Analytics** - Analyze ability usage patterns, economy decisions, and player behavior
- **Highlight Generation** - Auto-clip kills, clutches, and key moments based on event data
- **Machine Learning** - Train models for outcome prediction, player performance analysis, or tactical recommendations
- **Coaching Tools** - Review player states, ability cooldowns, and positioning frame-by-frame
- **Research** - Study meta evolution, agent pick rates, and strategic trends across tournaments

## How It Works

1. **VLR.gg Scraping** - Extract match metadata (teams, players, agents, starting sides)
2. **VOD Download** - Download YouTube VOD with optional timestamp ranges for individual maps
3. **Phase Detection** - Identify game phases: PREROUND → ACTIVE_ROUND → POST_ROUND
4. **HUD Analysis** - Extract information from spectator HUD regions:
   - Score, timer, spike status (OCR + template matching)
   - Agent detection (template matching with attack/defense variants)
   - Player states (health, armor, shields, abilities, ultimate)
   - Killfeed parsing (text recognition)
5. **Event Generation** - Fire timestamped events when state changes occur
6. **State Validation** - Validate events against game rules to prevent false positives
7. **Output** - Generate JSONL event logs and CSV frame states

## Project Structure

```
valoscribe/
├── src/valoscribe/
│   ├── commands/           # CLI command implementations
│   ├── config/             # HUD coordinate configs (champs2025.json)
│   ├── detectors/          # Computer vision detectors
│   │   ├── template_*.py   # Template matching (agents, health, armor, etc.)
│   │   ├── *_detector.py   # OCR-based detectors (killfeed, round, etc.)
│   │   └── cropper.py      # HUD region extraction
│   ├── orchestration/      # State management and event generation
│   │   ├── game_state_manager.py    # Main orchestration logic
│   │   ├── phase_detector.py        # Phase state machine
│   │   ├── round_manager.py         # Score and round tracking
│   │   ├── player_state_tracker.py  # Per-player state tracking
│   │   └── event_collector.py       # Event collection and output
│   ├── scraper/            # VLR.gg metadata scraper
│   ├── templates/          # Template images for matching
│   │   ├── preround_agents/         # Agent portraits (attack/defense)
│   │   ├── killfeed_agents/         # Killfeed agent icons
│   │   ├── score_digits/            # Score digit templates
│   │   ├── timer_digits/            # Timer digit templates
│   │   └── ...
│   ├── types/              # Pydantic models and type definitions
│   ├── utils/              # Logging, OCR utilities
│   └── video/              # YouTube download and frame reading
├── scripts/                # Batch processing scripts
│   ├── process_vlr_series.sh        # Process entire series from VLR URL
│   └── process_all_series_parallel.sh  # Parallel batch processing
└── tests/                  # Unit and integration tests
```

## Installation

### Prerequisites

- **Python 3.10+**
- **Tesseract OCR** - Required for killfeed text extraction
  ```bash
  # macOS
  brew install tesseract

  # Ubuntu/Debian
  sudo apt-get install tesseract-ocr

  # Windows
  # Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
  ```

- **yt-dlp** - Already included in dependencies for YouTube downloads

### Install Valoscribe

#### Option 1: Using uv (recommended)

[uv](https://github.com/astral-sh/uv) provides fast, reliable dependency management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/SphinxNumberNine/valoscribe.git
cd valoscribe

# Install dependencies
uv sync
```

#### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/SphinxNumberNine/valoscribe.git
cd valoscribe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Verify Installation

```bash
valoscribe --help
```

## Usage

### Quick Start: Process an Entire Series

The easiest way to process matches is using the VLR.gg integration:

```bash
# Process all maps from a VLR.gg match page
./scripts/process_vlr_series.sh https://www.vlr.gg/12345/team-a-vs-team-b

# This will:
# 1. Scrape match metadata from VLR.gg
# 2. Download VODs from YouTube
# 3. Process each map
# 4. Output structured event logs and frame states
```

### Individual Commands

#### 1. Scrape Match Metadata from VLR.gg

```bash
valoscribe scrape-vlr https://www.vlr.gg/12345/team-a-vs-team-b \
  --output ./output/metadata.json
```

Outputs team names, player names, agents, maps, VOD URLs, and starting sides.

#### 2. Download VOD from YouTube

```bash
valoscribe orchestrate download \
  --youtube-url "https://www.youtube.com/watch?v=..." \
  --output-dir ./vods \
  --start-time 00:15:30 \
  --end-time 01:45:20
```

Options:
- `--start-time` / `--end-time` - Extract specific map segments (optional)
- `--quality` - Video quality preference (default: 1080p)

#### 3. Process a VOD

```bash
valoscribe orchestrate process \
  --video-path ./vods/match.mp4 \
  --metadata-path ./output/metadata.json \
  --hud-config src/valoscribe/config/champs2025.json \
  --output-dir ./series_output/match_name/map1_ascent \
  --sample-rate 4
```

Key options:
- `--video-path` - Path to downloaded VOD
- `--metadata-path` - VLR.gg metadata JSON
- `--hud-config` - HUD coordinate configuration (use `champs2025.json` for 2025 Champions)
- `--output-dir` - Directory for outputs
- `--sample-rate` - Frames per second to process (default: 4)
- `--quiet` - Suppress progress output

This generates:
- `output/event_log.jsonl` - Timestamped game events
- `output/frame_states.csv` - Frame-by-frame player states

### Batch Processing

Process multiple matches in parallel:

```bash
# Create a file with VLR.gg URLs (one per line)
cat > matches.txt << EOF
https://www.vlr.gg/12345/team-a-vs-team-b
https://www.vlr.gg/12346/team-c-vs-team-d
https://www.vlr.gg/12347/team-e-vs-team-f
EOF

# Process all matches with 6 parallel jobs
./scripts/process_all_series_parallel.sh matches.txt 6
```

### Validate Outputs

Validate event logs for data quality:

```bash
./scripts/validate_event_logs.sh
```

Checks:
- Round start/end balance
- Match start/end presence
- Final score validity (>= 13, win by 2)

### Individual Detectors (Advanced)

Test individual detection components:

```bash
# Test agent detection
valoscribe detect agent \
  --video ./vods/match.mp4 \
  --frame 1000 \
  --player-index 0

# Test killfeed detection
valoscribe detect killfeed \
  --video ./vods/match.mp4 \
  --start-frame 5000 \
  --end-frame 5100

# Test score detection
valoscribe detect score \
  --video ./vods/match.mp4 \
  --frame 1000
```

## Development

### Running tests

```bash
uv run pytest
```

### Type checking

```bash
uv run mypy src/valoscribe
```

### Linting/formatting

```bash
uv run ruff check src/
uv run ruff format src/
```

## Output Format

Valoscribe generates two output files per map:

### 1. Event Log (`event_log.jsonl`)

JSONL format with one event per line. Each event is timestamped and includes relevant game state.

**Event Types:**

**Match Events:**
```json
{"type": "match_start", "timestamp": 0.0, "map_name": "Ascent", "team1": "Sentinels", "team2": "LOUD"}
{"type": "match_end", "timestamp": 2145.5, "winner": "Sentinels", "score_team1": 13, "score_team2": 8}
```

**Round Events:**
```json
{"type": "round_start", "timestamp": 15.2, "round_number": 1, "score_team1": 0, "score_team2": 0}
{"type": "round_end", "timestamp": 115.8, "round_number": 1, "score_team1": 1, "score_team2": 0, "winner": "Sentinels"}
```

**Kill Events:**
```json
{
  "type": "kill",
  "timestamp": 45.3,
  "killer": "TenZ",
  "killer_agent": "Jett",
  "victim": "aspas",
  "victim_agent": "Raze",
  "weapon": "Vandal",
  "headshot": true
}
```

**Ability Events:**
```json
{
  "type": "ability_used",
  "timestamp": 32.1,
  "player": "TenZ",
  "agent": "Jett",
  "ability_slot": "ability2",
  "ability_name": "Cloudburst"
}
```

**Ultimate Events:**
```json
{
  "type": "ultimate_used",
  "timestamp": 78.5,
  "player": "TenZ",
  "agent": "Jett",
  "ultimate_name": "Blade Storm"
}
```

**Spike Events:**
```json
{"type": "spike_plant", "timestamp": 95.2, "planter": "Sacy", "planter_agent": "Sova"}
```

### 2. Frame States (`frame_states.csv`)

CSV format with per-frame snapshots of all player states (sampled at 4 FPS by default).

**Columns:**
- `timestamp` - Video timestamp (seconds)
- `frame_number` - Frame index
- `phase` - Game phase (PREROUND, ACTIVE_ROUND, POST_ROUND)
- `round_number` - Current round
- `score_team1`, `score_team2` - Current scores
- For each player (0-9):
  - `player_{i}_name` - Player name
  - `player_{i}_agent` - Agent name
  - `player_{i}_team` - Team name
  - `player_{i}_health` - Current health (0-150)
  - `player_{i}_armor` - Current armor (0-50)
  - `player_{i}_shield` - Shield equipped (0=none, 1=light, 2=heavy)
  - `player_{i}_alive` - Alive status (True/False)
  - `player_{i}_ability1_available` - Ability 1 ready (True/False)
  - `player_{i}_ability2_available` - Ability 2 ready (True/False)
  - `player_{i}_ability3_available` - Ability 3 ready (True/False)
  - `player_{i}_ultimate_available` - Ultimate ready (True/False)
  - `player_{i}_ultimate_points` - Ultimate charge (0-8)
  - `player_{i}_credits` - Credits (economy)

**Example rows:**
```csv
timestamp,frame_number,phase,round_number,score_team1,score_team2,player_0_name,player_0_agent,...
15.25,183,ACTIVE_ROUND,1,0,0,TenZ,Jett,Sentinels,150,50,2,True,True,True,True,False,6,9000,...
15.50,186,ACTIVE_ROUND,1,0,0,TenZ,Jett,Sentinels,145,50,2,True,True,False,True,False,6,9000,...
```

### 3. Metadata (`metadata.json`)

Match metadata scraped from VLR.gg:

```json
{
  "match_id": "12345",
  "team1": "Sentinels",
  "team2": "LOUD",
  "map_name": "Ascent",
  "vod_url": "https://youtube.com/watch?v=...",
  "starting_side": {"Sentinels": "attack", "LOUD": "defense"},
  "players": {
    "Sentinels": ["TenZ", "Sacy", "pANcada", "Zekken", "johnqt"],
    "LOUD": ["aspas", "Less", "Saadhak", "cauanzin", "tuyz"]
  },
  "agents": {
    "TenZ": "Jett",
    "aspas": "Raze",
    ...
  }
}
```

## Configuration

### HUD Configs

HUD coordinate configurations define where to find UI elements in the video frame. Valoscribe includes a configuration for the 2025 Champions broadcast HUD:

**`src/valoscribe/config/champs2025.json`** - Champions 2025 broadcast HUD layout

If you need to process VODs with different HUD layouts:

1. Extract template coordinates using the `extract` commands:
   ```bash
   valoscribe extract agent-icon --video ./vods/match.mp4 --frame 1000 --player-index 0
   ```

2. Create a new config JSON with updated coordinates

3. Pass your config to the `orchestrate process` command:
   ```bash
   valoscribe orchestrate process --hud-config ./custom_config.json ...
   ```

### Supported Tournaments

Currently tested and validated on:
- **VCT Champions 2025** - Paris (using `champs2025.json` config)

The system should work with any 1080p Valorant broadcast that uses the standard spectator HUD, but coordinate configs may need adjustment for different production overlays.

## Limitations

- **1080p Resolution Only** - Template matching is calibrated for 1080p videos. Other resolutions may produce inaccurate detections.
- **Spectator HUD Required** - Player POV streams don't show all 10 players, so detection won't work.
- **Broadcast Interruptions** - Replays, technical pauses, and desk segments can cause false detections. The validation system mitigates most issues, but some edge cases remain.
- **Champions 2025 HUD** - Coordinate configs are specific to broadcast HUD layouts. New tournaments may require config updates.
- **Processing Time** - ~1.5-3 hours per match on a 14-core MacBook Pro. Cloud processing recommended for large batches.


## Agent Specific Limitations

- **Astra**: Ability usage data is not reliable. Astra's stars are notated differently on the HUD than all other abilities in the game, so I never got to handle the edge case. It's a TODO, will update this when solved.
- **Neon**: High Gear (Neon's Sprint) doesn't get marked on the HUD as "used" unless the sprint fuel hits 0. So Neon could be intermittently sprinting and there may not be events regarding it.
- **Chamber**: Tour de Force (Chamber's Ultimate) doesn't get marked on the HUD as "used" until he uses up all of his bullets or the round ends. So the event for this will likely be later than the true time the ultimate was popped.
- **Jett**: Bladestorm (Jett's Ultimate) doesn't get marked on the HUD as "used" until she uses up all of her blades or the round ends. So the event for this will likely be later than the true time the ultimate was popped.
- **Viper**: Toxic Screen and Poison Orb show up as "used" when Viper places the abilities. Turning off and re-activating them as fuel generates and depletes does not visually show on the HUD, and therefore is not possible to get events for.


## Known Issues

- **Round Start / End Mismatches** (~9/71 maps) - 9 of 71 processed maps for Champs 2025 have a mismatching number of round starts and round ends. This is likely caused by round replays / tech issues (verified for 3 of the 9 maps in question, assumed for the rest, YMMV)
- **Team Kills** - currently not supported, due to the way kill detection is coded. Happens very rarely, so not a huge priority for me to fix.
- **Preround Ability Usage** - hard to differentiate refunds from true ability placements, so abilities used during preround are not tracked. This is a TODO to fix for the future.

## Performance

**Champions 2025 Dataset (71 maps):**
- **62/71 maps** (87%) pass all validation checks
- **Event accuracy** - Round start/end balance, complete match events
- **Processing speed** - ~20-40 minutes per map (4 FPS sampling, 14-core MacBook Pro)
- **Average events** - 200-850 events per map
- **Frame state samples** - ~2,000-5,000 rows per map
- **Opening matches excluded** - the first 8 series of the event used a slightly different version of the HUD, and therefore were not processed in this first iteration. Working on processing these.

### Optional: ffmpeg decoding

By default frames are decoded with OpenCV, which decodes and converts every source frame even
though only one frame in `source_fps / 4` is analysed. Setting `VALOSCRIBE_FRAME_SOURCE=ffmpeg`
(and `VALOSCRIBE_FFMPEG` if `ffmpeg` is not on `PATH`) decodes through an ffmpeg subprocess
instead. In local testing on 1080p60 VODs this cut wall time by 1.5-2.8x with identical events on
5 of 6 clips; decoder pixel differences can move threshold-sensitive ability blips by one frame,
so OpenCV stays the default.

## Contributing

Contributions are welcome! Areas for improvement:

1. **HUD Configs** - Add support for new tournament broadcast layouts
2. **Agent Templates** - Update templates for new agents
3. **False Positive Reduction** - Improve phase detection to handle replays
4. **Performance** - Optimize detector performance for faster processing
5. **Documentation** - Improve usage examples and troubleshooting guides
6. **Testing** - Add more unit tests for edge cases

### Development Setup

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run type checking
uv run mypy src/valoscribe

# Format code
uv run ruff format src/
```

## Citation

If you use Valoscribe in your research or project, please cite:

```bibtex
@software{valoscribe2026,
  title = {Valoscribe: Automated Valorant VOD Analysis},
  author = {Krishnan, Ashwath},
  year = {2026},
  url = {https://github.com/SphinxNumberNine/valoscribe}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **VLR.gg** - Match metadata and statistics
- **Riot Games** - Valorant game and esports content
- **VCT Production** - High-quality broadcast spectator HUD

## Contact

For questions, issues, or feature requests, please open an issue on GitHub.
