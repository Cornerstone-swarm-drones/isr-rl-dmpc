from setuptools import setup
import os
from glob import glob

package_name = "isr_dmpc_sim"

setup(
    name=package_name,
    version="2.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
        (os.path.join("share", package_name, "meshes"), glob("meshes/*")),
        (os.path.join("share", package_name, "urdf"), glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Cornerstone-swarm-drones",
    maintainer_email="jrb252049@iitd.ac.in",
    description="ROS2 simulation package for the ISR-DMPC autonomous swarm system.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "swarm_dmpc_sim_node = isr_dmpc_sim.swarm_dmpc_sim_node:main",
            "rviz_bridge_node = isr_dmpc_sim.rviz_bridge_node:main",
            "hardware_bridge_node = isr_dmpc_sim.hardware_bridge_node:main",
        ],
    },
)
