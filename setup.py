from setuptools import setup, find_packages

setup(
    name="isr-dmpc",
    version="3.0.0",
    description="Autonomous multi-drone ISR swarm using MARL (MAPPO/SB3) + ADMM + DMPC (CVXPY/OSQP)",
    author="Cornerstone-swarm-drones",
    author_email="jrb252049@iitd.ac.in",
    url="https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=2.0.0",
        "pyyaml>=5.4",
        "cvxpy>=1.2.0",
        "osqp>=0.6.0",
    ],
    extras_require={
        "dev": [
            "tensorboard>=2.10.0",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "jupyter>=1.0.0",
            "matplotlib>=3.4.0",
        ],
    },
)