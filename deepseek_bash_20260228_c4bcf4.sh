#!/bin/bash
# scripts/install.sh - Professional installation script for Arch Linux

set -e

echo "========================================="
echo "Cup Tracker Pro - Professional Installation"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}Please do not run as root${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Checking system requirements..."

# Check Arch Linux
if [ ! -f /etc/arch-release ]; then
    echo -e "${RED}✗ This installer is for Arch Linux only${NC}"
    exit 1
fi

# Check for yay or pacman
if command -v yay &> /dev/null; then
    AUR_HELPER="yay"
elif command -v paru &> /dev/null; then
    AUR_HELPER="paru"
else
    echo -e "${YELLOW}⚠ No AUR helper found. Installing yay...${NC}"
    sudo pacman -S --needed git base-devel
    git clone https://aur.archlinux.org/yay.git
    cd yay
    makepkg -si
    cd ..
    rm -rf yay
    AUR_HELPER="yay"
fi

echo -e "${GREEN}✓${NC} Using AUR helper: $AUR_HELPER"

# Install dependencies
echo -e "${GREEN}✓${NC} Installing dependencies..."

# Core dependencies
sudo pacman -S --needed \
    python \
    python-pip \
    python-numpy \
    python-opencv \
    python-pyside6 \
    python-pillow \
    vulkan-intel \
    vulkan-radeon \
    vulkan-headers \
    glfw-wayland \
    cmake \
    ninja \
    base-devel

# AUR dependencies
$AUR_HELPER -S --needed \
    python-torch \
    python-torchvision \
    python-mss-git \
    python-pywayland \
    python-systemd \
    python-cuda \
    python-pycuda \
    opencl-amd \
    python-onnxruntime-gpu

echo -e "${GREEN}✓${NC} Dependencies installed"

# Create directories
echo -e "${GREEN}✓${NC} Creating directories..."
mkdir -p ~/.config/cup-tracker
mkdir -p ~/.local/share/cup-tracker/models
mkdir -p ~/.local/share/cup-tracker/logs
mkdir -p ~/.local/share/cup-tracker/patterns

# Clone and install
echo -e "${GREEN}✓${NC} Installing Cup Tracker Pro..."

# Build and install
cd /tmp
if [ -d "cup-tracker-pro" ]; then
    rm -rf cup-tracker-pro
fi

git clone https://github.com/yourrepo/cup-tracker-pro.git
cd cup-tracker-pro

# Build with optimizations
export CFLAGS="-march=native -O3 -pipe"
export CXXFLAGS="-march=native -O3 -pipe"

# Install Python package
pip install --user -e .

# Copy configuration
cp config/cup_tracker.conf ~/.config/cup-tracker/config.conf

# Download pretrained models
echo -e "${GREEN}✓${NC} Downloading pretrained models..."
python scripts/download_models.py --output ~/.local/share/cup-tracker/models/

# Setup systemd user service
echo -e "${GREEN}✓${NC} Setting up systemd service..."
mkdir -p ~/.config/systemd/user/
cp scripts/cup-tracker.service ~/.config/systemd/user/
systemctl --user daemon-reload

# Enable GPU optimizations
echo -e "${GREEN}✓${NC} Running GPU optimizations..."
bash scripts/optimize_gpu.sh

# Setup Vulkan layers
if command -v vulkaninfo &> /dev/null; then
    echo -e "${GREEN}✓${NC} Vulkan detected, configuring..."
    vulkaninfo --summary
fi

# Performance tuning
echo -e "${GREEN}✓${NC} Applying performance tuning..."

# CPU governor
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
fi

# GPU settings (NVIDIA)
if command -v nvidia-smi &> /dev/null; then
    sudo nvidia-smi -pm 1
    sudo nvidia-smi -ac 5001,1590
fi

# Create desktop entry
echo -e "${GREEN}✓${NC} Creating desktop entry..."
cat > ~/.local/share/applications/cup-tracker.desktop << EOF
[Desktop Entry]
Name=Cup Tracker Pro
Comment=Professional cup tracking system
Exec=$HOME/.local/bin/cup-tracker
Icon=$HOME/.local/share/icons/cup-tracker.png
Terminal=false
Type=Application
Categories=Game;Utility;
StartupNotify=true
EOF

# Copy icon
cp data/icon.png ~/.local/share/icons/cup-tracker.png

# Setup Wayland permissions (if needed)
if [ "$XDG_SESSION_TYPE" = "wayland" ]; then
    echo -e "${YELLOW}⚠ Wayland detected, setting permissions...${NC}"
    
    # Add user to necessary groups
    sudo usermod -aG video $USER
    sudo usermod -aG render $USER
    
    # Set CAP_SYS_NICE for real-time priority
    sudo setcap 'cap_sys_nice+ep' ~/.local/bin/cup-tracker
fi

# Verify installation
echo -e "${GREEN}✓${NC} Verifying installation..."
if python -c "import cup_tracker" 2>/dev/null; then
    echo -e "${GREEN}✓ Installation successful!${NC}"
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi

# Create uninstall script
cat > ~/.local/bin/cup-tracker-uninstall << 'EOF'
#!/bin/bash
echo "Uninstalling Cup Tracker Pro..."
rm -rf ~/.config/cup-tracker
rm -rf ~/.local/share/cup-tracker
rm -f ~/.local/bin/cup-tracker
rm -f ~/.local/share/applications/cup-tracker.desktop
systemctl --user disable cup-tracker.service
rm -f ~/.config/systemd/user/cup-tracker.service
echo "Uninstall complete"
EOF

chmod +x ~/.local/bin/cup-tracker-uninstall

echo ""
echo "========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================="
echo ""
echo "To start Cup Tracker Pro:"
echo "  $ cup-tracker"
echo ""
echo "To run with debugging:"
echo "  $ cup-tracker --debug"
echo ""
echo "To start as a service:"
echo "  $ systemctl --user start cup-tracker"
echo ""
echo "Configuration file: ~/.config/cup-tracker/config.conf"
echo "Logs directory: ~/.local/share/cup-tracker/logs"
echo ""
echo -e "${YELLOW}Note: You may need to log out and back in for all changes to take effect${NC}"