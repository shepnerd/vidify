from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="vidify",
    version="0.3.0",
    description="Video understanding agent — analyze, search, and edit videos with LLMs (parallel segment processing)",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "detection": ["ultralytics>=8.0.0"],
        "mra": [
            "opencv-python>=4.8.0",
            "Pillow>=10.0.0",
            "ultralytics>=8.0.0",
        ],
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"],
    },
    entry_points={
        'console_scripts': [
            'vidify=agent.main:cli',
        ],
    },
    python_requires=">=3.11",
    include_package_data=True,
)
