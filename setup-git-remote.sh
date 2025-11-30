#!/bin/bash
# Git Remote Setup for Nukem Fork

echo "==================================="
echo "Git Remote Setup for Nukem"
echo "==================================="
echo ""

# Check if fork exists
echo "Before running this script, make sure you have:"
echo "1. Created a fork of Haoming02/sd-webui-forge-classic on GitHub"
echo "2. Named it 'stable-diffusion-webui-Nukem' (or whatever you prefer)"
echo ""
read -p "Have you created your fork on GitHub? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Please create your fork first at:"
    echo "https://github.com/Haoming02/sd-webui-forge-classic"
    echo "Then run this script again."
    exit 1
fi

# Get the fork URL
read -p "Enter your GitHub username: " username
read -p "Enter your fork's repository name (default: stable-diffusion-webui-Nukem): " reponame
reponame=${reponame:-stable-diffusion-webui-Nukem}

fork_url="https://github.com/$username/$reponame.git"

echo ""
echo "Setting up remotes..."
echo "Fork URL: $fork_url"
echo ""

# Rename current origin to upstream
echo "1. Renaming 'origin' to 'upstream'..."
git remote rename origin upstream

# Add your fork as origin
echo "2. Adding your fork as 'origin'..."
git remote add origin "$fork_url"

# Verify
echo ""
echo "✓ Setup complete! Current remotes:"
git remote -v

echo ""
echo "==================================="
echo "Next steps:"
echo "==================================="
echo ""
echo "1. Push your changes to your fork:"
echo "   git push -u origin neo"
echo ""
echo "2. To get updates from the original Forge Classic:"
echo "   git fetch upstream"
echo "   git merge upstream/neo"
echo ""
echo "3. To push updates to your fork:"
echo "   git push origin neo"
echo ""
