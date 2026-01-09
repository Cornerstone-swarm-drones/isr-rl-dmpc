from setuptools import setup, find_packages

setup(
    name="isr-rl-dmpc",
    version="0.1.0",
    description="Autonomous swarm with RL-based DMPC",
    author="Cornerstone-swarm-drones",
    author_email="jrb252049@iitd.ac.in",
    url="https://github.com/Cornerstone-swarm-drones/isr-rl-dmpc.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "gym>=0.21.0",
        "pyyaml>=5.4",
        "pytest>=6.2.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
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