# ISR-RL-DMPC: Autonomous Swarm with RL-Based DMPC

Grid-based Intelligence, Surveillance & Reconnaissance system with distributed 
multi-agent control via Reinforcement Learning and Model Predictive Control.

## Quick Start

```bash
# Clone repository
git clone https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git
cd isr-rl-dmpc

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start notebook
jupyter notebook notebooks/01_quickstart.ipynb
```

## Project Structure
- src/isr_rl_dmpc/ - Main package code
- tests/ - Unit and integration tests
- config/ - Configuration files
- notebooks/ - Jupyter tutorials
- scripts/ - Training and evaluation scripts
- docs/ - Documentation

## Team
- Jivesh Kesar: jrb252026@iitd.ac.in
- Harsh: jrb252049@iitd.ac.in
- Rohit Shankar Sinha: jrb252051@iitd.ac.in
