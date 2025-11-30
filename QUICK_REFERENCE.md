# Quick Reference - Nukem WebUI

## Essential Commands

```bash
# Interactive Development Menu
./dev-tools.sh

# Launch WebUI
./webui-user.sh

# Or via dev tools
./dev-tools.sh launch
```

## Development Tools Menu

```
./dev-tools.sh

1) Setup virtual environment     - First time setup
2) Install dependencies          - Install Python packages
3) Run basic tests              - Verify installation
4) Quick launch (dev mode)      - Start the WebUI
5) System information           - Check GPU, Python, disk
6) Clean build artifacts        - Remove cache files
7) Create backup                - Backup your work
8) Update from upstream         - Get Forge Classic updates
9) Exit
```

## One-Line Commands

```bash
./dev-tools.sh setup     # Create venv
./dev-tools.sh install   # Install deps
./dev-tools.sh test      # Run tests
./dev-tools.sh launch    # Launch WebUI
./dev-tools.sh info      # System info
./dev-tools.sh clean     # Clean cache
./dev-tools.sh backup    # Backup code
./dev-tools.sh update    # Update upstream
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes, then commit
git add .
git commit -m "feat: description"

# Push to your fork
git push origin feature/my-feature

# Get updates from Forge Classic
git fetch haoming
git merge haoming/neo
```

## URLs

- **WebUI:** http://localhost:7860
- **API Docs:** http://localhost:7860/docs
- **Your Fork:** https://github.com/ivanuser/stable-diffusion-webui-Nukem

## Project Structure

```
stable-diffusion-webui-Nukem/
├── backend/              # Core engine (add models here)
│   ├── huggingface/     # Model implementations
│   └── loader.py        # Model detection
├── modules/             # UI & processing
├── modules_forge/       # Forge features
├── extensions/          # Add extensions here
└── models/              # Put model files here
    ├── Stable-diffusion/
    ├── Lora/
    └── ControlNet/
```

## Configuration

Edit `webui-user.sh`:

```bash
# Enable API
export COMMANDLINE_ARGS="--api"

# Performance (RTX 30+)
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --xformers --cuda-malloc"

# Fastest (SageAttention)
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --sage"

# Low VRAM
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --medvram"
```

## Troubleshooting

```bash
# Clean and restart
./dev-tools.sh clean
rm -rf venv
./dev-tools.sh setup
./dev-tools.sh install

# Check system
./dev-tools.sh info

# View logs
tail -f logs/webui.log  # if exists
```

## Documentation

- `SETUP_COMPLETE.md` - Setup summary & next steps
- `DEVELOPMENT.md` - Full development guide
- `CLAUDE.md` - Architecture reference
- `ANALYSIS.md` - Feature roadmap
- `PROJECT_STATUS.md` - Current status

## Help

Need help? Check:
1. `DEVELOPMENT.md` for detailed guides
2. Console output for error messages
3. GitHub issues (once you encounter problems)

---

**Quick Start:** `./dev-tools.sh` → Option 1 → Option 2 → Option 4
