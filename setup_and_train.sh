#!/bin/bash
set -e

# ==========================================
# ISR-RL-DMPC Brev Deployment Script
# ==========================================

REPO_URL="https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git"
REPO_DIR="isr-rl-dmpc"
ENV_NAME="isr"

echo "🚀 Starting ISR-RL-DMPC setup on NVIDIA Brev..."

# 1. Clone the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "📦 Cloning repository..."
    git clone $REPO_URL
else
    echo "✅ Repository already exists. Pulling latest changes..."
    cd $REPO_DIR && git pull && cd ..
fi

cd $REPO_DIR

# 2. Check for Conda and install Miniconda if missing
if ! command -v conda &> /dev/null; then
    echo "⚙️ Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "✅ Conda is already installed."
fi

# 3. Setup the Conda Environment
echo "🐍 Creating Conda environment from environment.yml..."
# Remove existing environment if it exists to ensure a clean slate
conda env remove -n $ENV_NAME -y || true
conda env create -f environment.yml -n $ENV_NAME

# Activate the environment
echo "🔄 Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# 4. Verify GPU availability in the environment
echo "🖥️ Checking PyTorch GPU availability..."
python -c "import torch; print(f'PyTorch CUDA Available: {torch.cuda.is_available()}')"

# 5. Pipeline Execution
echo "▶️ Starting Pipeline Execution..."

# Ensure necessary directories exist
mkdir -p data/training_logs
mkdir -p data/trained_models
mkdir -p config

# Step 5a: Calibrate Sensors (if required by your workflow)
echo "📡 Running Sensor Calibration..."
python scripts/calibrate_sensors.py

# Step 5b: Train the Agent
# Vectorized environments are recommended for RL to speed up training
echo "🧠 Training the RL Agent..."
python scripts/train_agent.py --config config/default_config.yaml --episodes 500 --steps 1000

# Step 5c: Evaluate the Policy
echo "📊 Evaluating the Policy..."
# Assuming the latest checkpoint is saved; adjust path if necessary based on train_agent.py output
LATEST_CHECKPOINT=$(ls -t data/training_logs/*/checkpoint_ep*.pt | head -n 1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    python scripts/test_mission.py --checkpoint "$LATEST_CHECKPOINT" --episodes 100
else
    echo "⚠️ No checkpoint found to evaluate."
fi

# Step 5d: Visualize Results
echo "📈 Generating Visualizations..."
python scripts/visualize_results.py

echo "🎉 ISR-RL-DMPC Pipeline Completed Successfully!"
