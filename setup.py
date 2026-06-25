from setuptools import setup, find_packages

def _read_requirements(path):
    with open(path) as f:
        return [
            line.strip() for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-r ")
        ]


requirements = _read_requirements("requirements.txt")

extras = {
    "serving": ["vllm>=0.19.0"],
    "direct": ["vllm>=0.19.0", "transformers>=4.40.0"],
    "asr": [
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "librosa>=0.10.0",
    ],
    "ocr": [
        "opencv-python>=4.8.0",
        "paddlepaddle>=2.6.0",
        "paddleocr>=2.7.0",
        "pytesseract>=0.3.10",
    ],
    "emotion": [
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "librosa>=0.10.0",
        "fer>=22.0.0",
        "opencv-python>=4.8.0",
    ],
    "live": ["opencv-python>=4.8.0"],
    "detection": ["opencv-python>=4.8.0", "ultralytics>=8.0.0"],
    "mra": [
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "ultralytics>=8.0.0",
    ],
    "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"],
}

extras["full"] = sorted({
    dep
    for group in ("serving", "direct", "asr", "ocr", "emotion", "live", "detection", "mra")
    for dep in extras[group]
})
extras["all"] = extras["full"]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vidify",
    version="0.3.0",
    description="Video understanding agent — analyze, search, and edit videos with LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'vidify=agent.main:cli',
        ],
    },
    python_requires=">=3.11",
    include_package_data=True,
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
