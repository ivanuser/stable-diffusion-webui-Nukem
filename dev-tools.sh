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
        print_info "Activate it with: source venv/bin/activate"
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

    print_info "Activating venv..."
    source venv/bin/activate

    print_info "Upgrading pip..."
    pip install --upgrade pip

    print_info "Installing requirements..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        return 1
    fi

    print_success "Dependencies installed"
}

#############################
# Run Tests
#############################
function run_tests() {
    print_header "Running Tests"

    # Check if venv exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi

    # Basic import test
    print_info "Testing Python imports..."
    python3 -c "
import sys
try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError:
    print('✗ PyTorch not installed')
    sys.exit(1)

try:
    import gradio
    print('✓ Gradio:', gradio.__version__)
except ImportError:
    print('✗ Gradio not installed')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        print_success "Basic imports working"
    else
        print_error "Import test failed"
        return 1
    fi
}

#############################
# Quick Launch
#############################
function quick_launch() {
    print_header "Quick Launch (Development Mode)"

    print_info "Starting WebUI..."
    print_info "Press Ctrl+C to stop"
    echo ""

    ./webui-user.sh
}

#############################
# Check System Info
#############################
function system_info() {
    print_header "System Information"

    echo -e "${BLUE}Python:${NC}"
    python3 --version

    echo ""
    echo -e "${BLUE}GPU:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "No NVIDIA GPU detected (or nvidia-smi not found)"
    fi

    echo ""
    echo -e "${BLUE}Disk Space (this directory):${NC}"
    du -sh .

    echo ""
    echo -e "${BLUE}Virtual Environment:${NC}"
    if [ -d "venv" ]; then
        echo "✓ Found at ./venv"
        if [ -f "venv/bin/python" ]; then
            venv/bin/python --version
        fi
    else
        echo "✗ Not found"
    fi

    echo ""
    echo -e "${BLUE}Git Status:${NC}"
    git status -s
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

    print_info "Fetching latest changes from haoming/neo..."
    git fetch haoming

    print_info "Current branch:"
    git branch --show-current

    echo ""
    print_info "New commits available:"
    git log --oneline HEAD..haoming/neo | head -10

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
# Main Menu
#############################
function show_menu() {
    clear
    print_header "Nukem WebUI - Development Tools"
    echo ""
    echo "1) Setup virtual environment"
    echo "2) Install dependencies"
    echo "3) Run basic tests"
    echo "4) Quick launch (dev mode)"
    echo "5) System information"
    echo "6) Clean build artifacts"
    echo "7) Create backup"
    echo "8) Update from upstream (Haoming02)"
    echo "9) Exit"
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
        9) exit 0 ;;
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
        *)
            echo "Usage: $0 {setup|install|test|launch|info|clean|backup|update}"
            echo ""
            echo "Or run without arguments for interactive menu"
            exit 1
            ;;
    esac
fi
