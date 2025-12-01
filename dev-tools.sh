#!/bin/bash
#########################################################
# Development Tools for Nukem WebUI
#########################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

#############################
# Get Python Command
#############################
function get_python() {
    # Use venv python if available, otherwise system python
    if [ -f "venv/bin/python" ]; then
        echo "venv/bin/python"
    elif [ -f "venv/Scripts/python.exe" ]; then
        echo "venv/Scripts/python.exe"
    else
        echo "python3"
    fi
}

function get_pip() {
    # Use venv pip if available
    if [ -f "venv/bin/pip" ]; then
        echo "venv/bin/pip"
    elif [ -f "venv/Scripts/pip.exe" ]; then
        echo "venv/Scripts/pip.exe"
    else
        echo "pip3"
    fi
}

#############################
# Setup Virtual Environment
#############################
function setup_venv() {
    print_header "Setting Up Virtual Environment"

    if [ -d "venv" ]; then
        print_info "Virtual environment already exists at ./venv"
        read -p "Recreate it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing old venv..."
            rm -rf venv
        else
            print_success "Using existing venv"
            return 0
        fi
    fi

    print_info "Creating virtual environment..."
    python3 -m venv venv

    if [ $? -eq 0 ]; then
        print_success "Virtual environment created at ./venv"

        # Upgrade pip immediately
        print_info "Upgrading pip..."
        $(get_pip) install --upgrade pip

        print_info "To activate manually: source venv/bin/activate"
        return 0
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

#############################
# Install Dependencies
#############################
function install_deps() {
    print_header "Installing Dependencies"

    if [ ! -d "venv" ]; then
        print_error "Virtual environment not found. Run setup first."
        return 1
    fi

    local PIP=$(get_pip)
    local PYTHON=$(get_python)

    print_info "Using pip: $PIP"
    print_info "Using python: $PYTHON"

    print_info "Upgrading pip..."
    $PIP install --upgrade pip

    # Install PyTorch first (if not already installed)
    print_info "Checking PyTorch..."
    if ! $PYTHON -c "import torch" 2>/dev/null; then
        print_info "Installing PyTorch (this may take a while)..."
        # Check if CUDA is available
        if command -v nvidia-smi &> /dev/null; then
            print_info "NVIDIA GPU detected, installing CUDA version..."
            $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        else
            print_info "No NVIDIA GPU detected, installing CPU version..."
            $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_success "PyTorch already installed"
    fi

    # Install Gradio with compatible version
    print_info "Installing Gradio (compatible version)..."
    $PIP install 'gradio==4.44.1' 'gradio-client==1.3.0' gradio_rangeslider

    # Install main requirements
    print_info "Installing requirements from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        $PIP install -r requirements.txt
    else
        print_error "requirements.txt not found"
        return 1
    fi

    print_success "Dependencies installed"
    echo ""
    print_info "Run './dev-tools.sh test' to verify installation"
}

#############################
# Run Tests
#############################
function run_tests() {
    print_header "Running Tests"

    local PYTHON=$(get_python)

    if [ "$PYTHON" == "python3" ] && [ -d "venv" ]; then
        print_error "Virtual environment exists but not being used."
        print_info "The venv python was not found at expected location."
        print_info "Try running: source venv/bin/activate && python3 -c 'import torch'"
    fi

    print_info "Using Python: $PYTHON"
    echo ""

    # Basic import test
    print_info "Testing Python imports..."

    $PYTHON -c "
import sys
print('Python:', sys.version.split()[0])
print('Path:', sys.executable)
print('')

# Test PyTorch
try:
    import torch
    print('✓ PyTorch:', torch.__version__)
    if torch.cuda.is_available():
        print('  CUDA available:', torch.cuda.get_device_name(0))
    else:
        print('  CUDA: Not available (CPU mode)')
except ImportError as e:
    print('✗ PyTorch not installed:', e)
    sys.exit(1)

# Test Gradio
try:
    import gradio
    print('✓ Gradio:', gradio.__version__)
except ImportError as e:
    print('✗ Gradio not installed:', e)
    sys.exit(1)

# Test Transformers
try:
    import transformers
    print('✓ Transformers:', transformers.__version__)
except ImportError as e:
    print('⚠ Transformers not installed (optional):', e)

# Test other key packages
try:
    import safetensors
    print('✓ Safetensors:', safetensors.__version__)
except ImportError:
    print('⚠ Safetensors not installed')

try:
    import PIL
    print('✓ Pillow:', PIL.__version__)
except ImportError:
    print('⚠ Pillow not installed')

print('')
print('All core imports successful!')
"

    if [ $? -eq 0 ]; then
        echo ""
        print_success "Basic imports working"
    else
        echo ""
        print_error "Import test failed"
        print_info "Try running: ./dev-tools.sh install"
        return 1
    fi
}

#############################
# Quick Launch
#############################
function quick_launch() {
    print_header "Quick Launch (Development Mode)"

    local PYTHON=$(get_python)

    # Check if dependencies are installed
    if ! $PYTHON -c "import torch; import gradio" 2>/dev/null; then
        print_error "Dependencies not installed!"
        print_info "Run './dev-tools.sh install' first"
        return 1
    fi

    print_info "Starting WebUI..."
    print_info "Press Ctrl+C to stop"
    echo ""

    # Detect GPU and set appropriate args
    local EXTRA_ARGS=""
    if command -v nvidia-smi &> /dev/null; then
        print_info "NVIDIA GPU detected"
        EXTRA_ARGS="--xformers"
    else
        print_info "No GPU detected, using CPU mode (slow!)"
        EXTRA_ARGS="--always-cpu --skip-torch-cuda-test"
    fi

    # Launch with appropriate settings
    $PYTHON launch.py --skip-python-version-check --listen --api $EXTRA_ARGS
}

#############################
# Check System Info
#############################
function system_info() {
    print_header "System Information"

    local PYTHON=$(get_python)

    echo -e "${BLUE}Python:${NC}"
    $PYTHON --version 2>/dev/null || python3 --version
    echo "  Location: $PYTHON"

    echo ""
    echo -e "${BLUE}GPU:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    else
        echo "  No NVIDIA GPU detected (or nvidia-smi not found)"
    fi

    echo ""
    echo -e "${BLUE}CUDA (via PyTorch):${NC}"
    $PYTHON -c "
import torch
if torch.cuda.is_available():
    print('  Available: Yes')
    print('  Version:', torch.version.cuda)
    print('  Device:', torch.cuda.get_device_name(0))
else:
    print('  Available: No')
" 2>/dev/null || echo "  PyTorch not installed"

    echo ""
    echo -e "${BLUE}Disk Space (this directory):${NC}"
    du -sh . 2>/dev/null || echo "  Unable to check"

    echo ""
    echo -e "${BLUE}Virtual Environment:${NC}"
    if [ -d "venv" ]; then
        echo "  ✓ Found at ./venv"
        if [ -f "venv/bin/python" ]; then
            echo "  Python: $(venv/bin/python --version 2>&1)"
        elif [ -f "venv/Scripts/python.exe" ]; then
            echo "  Python: $(venv/Scripts/python.exe --version 2>&1)"
        fi
    else
        echo "  ✗ Not found - run './dev-tools.sh setup'"
    fi

    echo ""
    echo -e "${BLUE}Installed Packages:${NC}"
    if [ -d "venv" ]; then
        $PYTHON -c "
try:
    import torch; print('  PyTorch:', torch.__version__)
except: print('  PyTorch: Not installed')
try:
    import gradio; print('  Gradio:', gradio.__version__)
except: print('  Gradio: Not installed')
try:
    import transformers; print('  Transformers:', transformers.__version__)
except: print('  Transformers: Not installed')
" 2>/dev/null
    else
        echo "  (venv not set up)"
    fi

    echo ""
    echo -e "${BLUE}Git Status:${NC}"
    git status -s 2>/dev/null || echo "  Not a git repository"
}

#############################
# Clean Build Artifacts
#############################
function clean() {
    print_header "Cleaning Build Artifacts"

    print_info "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    find . -type f -name "*.pyc" -delete 2>/dev/null
    find . -type f -name "*.pyo" -delete 2>/dev/null

    print_info "Removing build directories..."
    rm -rf build dist *.egg-info

    print_info "Removing log files..."
    rm -f webui.log *.log

    print_success "Cleanup complete"
}

#############################
# Backup Current State
#############################
function backup() {
    print_header "Creating Backup"

    BACKUP_NAME="nukem-backup-$(date +%Y%m%d-%H%M%S).tar.gz"

    print_info "Creating backup: $BACKUP_NAME"
    print_info "Excluding: venv, models, outputs, .git"

    tar -czf "../$BACKUP_NAME" \
        --exclude="venv" \
        --exclude="models" \
        --exclude="outputs" \
        --exclude=".git" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude="*.log" \
        .

    if [ $? -eq 0 ]; then
        print_success "Backup created: ../$BACKUP_NAME"
    else
        print_error "Backup failed"
    fi
}

#############################
# Update from Upstream
#############################
function update_upstream() {
    print_header "Update from Haoming02's Forge Classic"

    # Check if haoming remote exists
    if ! git remote | grep -q "haoming"; then
        print_info "Adding haoming remote..."
        git remote add haoming https://github.com/Haoming02/sd-webui-forge-classic.git
    fi

    print_info "Fetching latest changes from haoming/neo..."
    git fetch haoming

    print_info "Current branch:"
    git branch --show-current

    echo ""
    print_info "New commits available:"
    git log --oneline HEAD..haoming/neo 2>/dev/null | head -10

    if [ $(git log --oneline HEAD..haoming/neo 2>/dev/null | wc -l) -eq 0 ]; then
        print_success "Already up to date!"
        return 0
    fi

    echo ""
    read -p "Merge these changes? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git merge haoming/neo
        if [ $? -eq 0 ]; then
            print_success "Merge successful"
        else
            print_error "Merge conflict - resolve manually"
        fi
    else
        print_info "Skipped merge"
    fi
}

#############################
# Full Setup (setup + install + test)
#############################
function full_setup() {
    print_header "Full Setup (venv + dependencies + test)"

    setup_venv
    if [ $? -ne 0 ]; then
        print_error "Setup failed"
        return 1
    fi

    echo ""
    install_deps
    if [ $? -ne 0 ]; then
        print_error "Install failed"
        return 1
    fi

    echo ""
    run_tests
    if [ $? -ne 0 ]; then
        print_error "Tests failed"
        return 1
    fi

    echo ""
    print_success "Full setup complete! Run './dev-tools.sh launch' to start."
}

#############################
# Main Menu
#############################
function show_menu() {
    clear
    print_header "Nukem WebUI - Development Tools"
    echo ""
    echo "  1) Setup virtual environment"
    echo "  2) Install dependencies"
    echo "  3) Run basic tests"
    echo "  4) Quick launch (dev mode)"
    echo "  5) System information"
    echo "  6) Clean build artifacts"
    echo "  7) Create backup"
    echo "  8) Update from upstream (Haoming02)"
    echo "  9) Full setup (1+2+3 combined)"
    echo "  0) Exit"
    echo ""
    read -p "Select option: " choice

    case $choice in
        1) setup_venv ;;
        2) install_deps ;;
        3) run_tests ;;
        4) quick_launch ;;
        5) system_info ;;
        6) clean ;;
        7) backup ;;
        8) update_upstream ;;
        9) full_setup ;;
        0) exit 0 ;;
        *) print_error "Invalid option" ;;
    esac

    echo ""
    read -p "Press enter to continue..."
    show_menu
}

#############################
# Command Line Args
#############################
if [ $# -eq 0 ]; then
    show_menu
else
    case $1 in
        setup) setup_venv ;;
        install) install_deps ;;
        test) run_tests ;;
        launch) quick_launch ;;
        info) system_info ;;
        clean) clean ;;
        backup) backup ;;
        update) update_upstream ;;
        full) full_setup ;;
        *)
            echo "Usage: $0 {setup|install|test|launch|info|clean|backup|update|full}"
            echo ""
            echo "Commands:"
            echo "  setup   - Create virtual environment"
            echo "  install - Install all dependencies"
            echo "  test    - Run basic import tests"
            echo "  launch  - Start the WebUI"
            echo "  info    - Show system information"
            echo "  clean   - Clean build artifacts"
            echo "  backup  - Create backup archive"
            echo "  update  - Update from upstream Forge Classic"
            echo "  full    - Full setup (setup + install + test)"
            echo ""
            echo "Or run without arguments for interactive menu"
            exit 1
            ;;
    esac
fi
