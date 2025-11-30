# Development Guide - Nukem WebUI

## Quick Start

### 1. Initial Setup

```bash
# Run the development tools menu
./dev-tools.sh

# Or use individual commands:
./dev-tools.sh setup    # Create virtual environment
./dev-tools.sh install  # Install dependencies
./dev-tools.sh test     # Run basic tests
./dev-tools.sh launch   # Start the WebUI
```

### 2. First Launch

```bash
# Simple launch
./webui-user.sh

# Or with the dev tools
./dev-tools.sh launch
```

The WebUI will be available at: `http://localhost:7860`

API documentation: `http://localhost:7860/docs`

## Development Environment

### Virtual Environment

The project uses Python virtual environments to isolate dependencies:

```bash
# Create venv (Python 3.11+ recommended, 3.12 should work)
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate when done
deactivate
```

### Environment Configuration

Edit `webui-user.sh` (Linux/macOS) or `webui-user.bat` (Windows) to customize:

**Essential Settings:**
```bash
# Enable API for development
export COMMANDLINE_ARGS="--api"

# Allow network access
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --listen"

# Custom port
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --port 7860"
```

**Performance Settings:**
```bash
# For RTX 30+ GPUs
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --xformers --cuda-malloc --cuda-stream"

# For fastest generation (SageAttention)
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --sage"
```

**Development Speedup (after first successful launch):**
```bash
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --skip-prepare-environment --skip-install"
```

## Development Tools

### dev-tools.sh Menu

Interactive menu with common tasks:

```bash
./dev-tools.sh

Options:
1) Setup virtual environment    - Create venv
2) Install dependencies         - Install packages
3) Run basic tests             - Test imports
4) Quick launch (dev mode)     - Start WebUI
5) System information          - Check GPU, Python, disk
6) Clean build artifacts       - Remove cache files
7) Create backup               - Backup code (excludes models)
8) Update from upstream        - Merge Haoming02's changes
9) Exit
```

### Command Line Usage

```bash
./dev-tools.sh setup     # Setup venv
./dev-tools.sh install   # Install deps
./dev-tools.sh test      # Run tests
./dev-tools.sh launch    # Launch WebUI
./dev-tools.sh info      # System info
./dev-tools.sh clean     # Clean cache
./dev-tools.sh backup    # Create backup
./dev-tools.sh update    # Update from upstream
```

## Development Workflow

### Making Changes

1. **Create a feature branch:**
```bash
git checkout -b feature/my-feature
```

2. **Make your changes**
   - Edit files in `modules/`, `backend/`, or `extensions/`
   - Test frequently with `./webui-user.sh`

3. **Test your changes:**
```bash
# Quick test - just run the UI
./dev-tools.sh launch

# Test imports
./dev-tools.sh test

# Check for Python errors
python3 -m py_compile path/to/your/file.py
```

4. **Commit changes:**
```bash
git add .
git commit -m "feat: add new feature X"
```

5. **Push to your fork:**
```bash
git push origin feature/my-feature
```

### Commit Message Convention

Use conventional commits format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add AnimateDiff motion module support
fix: resolve VRAM leak in model switching
docs: update DEVELOPMENT.md with testing instructions
refactor: reorganize backend loader code
```

### Syncing with Upstream

Stay updated with Haoming02's Forge Classic:

```bash
# Fetch latest changes
git fetch haoming

# View what's new
git log HEAD..haoming/neo

# Merge updates
git merge haoming/neo

# Or use the dev tool
./dev-tools.sh update
```

Resolve conflicts if any, then:
```bash
git add .
git commit -m "merge: sync with haoming/neo"
git push origin neo
```

## Testing

### Manual Testing Checklist

Before committing major changes, test:

- [ ] WebUI launches without errors
- [ ] Can load a model (FLUX, SDXL, etc.)
- [ ] Can generate an image (txt2img)
- [ ] Can generate with img2img
- [ ] Extensions load properly
- [ ] API endpoints work (if modified)
- [ ] No VRAM leaks (check with `nvidia-smi`)

### Testing New Models

```bash
# 1. Place model in models/Stable-diffusion/
cp /path/to/model.safetensors models/Stable-diffusion/

# 2. Launch WebUI
./webui-user.sh

# 3. Select model in UI
# 4. Generate test image
# 5. Check console for errors
```

### Testing Extensions

```bash
# 1. Create extension directory
mkdir extensions/my-extension

# 2. Add your extension files
# extensions/my-extension/
#   ├── scripts/
#   │   └── my_script.py
#   └── install.py (optional)

# 3. Restart WebUI
# Extension should appear in UI
```

### API Testing

With `--api` flag enabled:

```bash
# Test txt2img endpoint
curl -X POST http://localhost:7860/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat",
    "steps": 20,
    "width": 512,
    "height": 512
  }'

# View API docs
# Open http://localhost:7860/docs in browser
```

## Debugging

### Enable Debug Logging

Add to `webui-user.sh`:
```bash
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --debug"
```

### Common Issues

**Issue: Out of Memory (CUDA)**
```bash
# Solution: Use lower VRAM mode
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --medvram"
# or
export COMMANDLINE_ARGS="$COMMANDLINE_ARGS --lowvram"
```

**Issue: Model not detected**
- Check filename/path patterns in `backend/loader.py`
- Add debug prints to see what's being detected
- Check model config files

**Issue: Import errors**
```bash
# Reinstall dependencies
./dev-tools.sh install

# Or manually:
source venv/bin/activate
pip install -r requirements.txt
```

**Issue: Port already in use**
```bash
# Use different port
export COMMANDLINE_ARGS="--port 7861"
```

### Debug Workflow

1. **Enable verbose output:**
```python
# Add to your code
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check console output:**
   - WebUI prints detailed logs during startup
   - Watch for errors during model loading
   - Monitor VRAM usage

3. **Use Python debugger:**
```python
# Add breakpoint in your code
import pdb; pdb.set_trace()
```

4. **Check backend logs:**
```bash
# Watch logs in real-time
tail -f venv/lib/python*/site-packages/torch/*.log
```

## Performance Optimization

### Benchmarking

Test generation speed:

```bash
# Generate same image multiple times
# Note the time in console output
# Compare before/after changes
```

### Memory Profiling

```bash
# Watch VRAM usage
watch -n 1 nvidia-smi

# Or use Python profiler
python3 -m memory_profiler your_script.py
```

## Code Style

### Python

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions
- Keep functions small and focused

```python
def my_function(param: str) -> dict:
    """
    Brief description of function.

    Args:
        param: Description of parameter

    Returns:
        Description of return value
    """
    # Implementation
    return {}
```

### Formatting

```bash
# Format code with black (if installed)
black your_file.py

# Or use ruff
ruff format your_file.py
```

## Project Structure for Development

### Key Directories

**Backend (Core Engine):**
```
backend/
├── huggingface/        # Add new model implementations here
├── diffusion_engine/   # Modify sampling/diffusion logic
├── nn/                 # Neural network layers
├── loader.py          # Modify model detection/loading
└── attention.py       # Attention mechanisms
```

**Modules (UI & Processing):**
```
modules/
├── ui.py              # Main Gradio UI
├── processing.py      # Image processing pipeline
├── sd_models.py       # Model management
└── ...                # Various utilities
```

**Modules Forge (Forge-Specific):**
```
modules_forge/
├── main_entry.py      # Main processing entry point
├── presets.py         # Workflow presets
└── ...
```

**Extensions:**
```
extensions/
└── your-extension/    # Add your extension here
    ├── scripts/
    │   └── my_script.py
    ├── install.py
    └── README.md
```

### Adding a New Feature

**Example: Adding AnimateDiff**

1. **Backend implementation:**
```bash
# Create model class
touch backend/huggingface/guoyww/animatediff.py

# Add detection to loader
# Edit: backend/loader.py
```

2. **UI integration:**
```bash
# Create UI module
touch modules/animatediff_ui.py

# Or create as extension
mkdir -p extensions/animatediff/scripts
touch extensions/animatediff/scripts/animatediff_script.py
```

3. **Test:**
```bash
./webui-user.sh
# Test in UI
```

4. **Commit:**
```bash
git add .
git commit -m "feat: add AnimateDiff support"
git push origin feature/animatediff
```

## Useful Commands

```bash
# Check Python imports work
python3 -c "import torch; print(torch.__version__)"
python3 -c "import gradio; print(gradio.__version__)"

# Find large files (models)
du -h models/ | sort -rh | head -20

# Check disk space
df -h .

# Monitor GPU
watch -n 1 nvidia-smi

# Kill process on port 7860
lsof -ti:7860 | xargs kill -9

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Search for code
grep -r "pattern" --include="*.py"

# View commit history
git log --oneline --graph --all

# View file changes
git diff path/to/file.py
```

## Resources

### Documentation
- Forge Neo README: `README.md`
- Project roadmap: `ANALYSIS.md`
- AI assistant guide: `CLAUDE.md`
- Project status: `PROJECT_STATUS.md`

### External Resources
- Forge Classic: https://github.com/Haoming02/sd-webui-forge-classic
- Original Forge: https://github.com/lllyasviel/stable-diffusion-webui-forge
- A1111 Wiki: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki
- Gradio Docs: https://gradio.app/docs
- PyTorch Docs: https://pytorch.org/docs

## Getting Help

1. Check existing documentation
2. Search closed issues on GitHub
3. Ask in project Discord/forum (once established)
4. Create a detailed issue on GitHub

When reporting bugs, include:
- OS and Python version
- GPU model and VRAM
- Console output/error messages
- Steps to reproduce
- What you expected vs. what happened

---

**Happy coding! 🚀**

*Last updated: 2025-11-30*
