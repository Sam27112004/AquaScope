from setuptools import setup, find_packages

setup(
    name="aquascope",
    version="0.1.0",
    description="AI-based underwater inspection analysis system",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.26.0",
        "opencv-python>=4.9.0",
        "PyYAML>=6.0.1",
        "pydantic>=2.6.0",
        "loguru>=0.7.2",
    ],
    entry_points={
        "console_scripts": [
            "aq-train=scripts.train:main",
            "aq-infer=scripts.infer_video:main",
            "aq-eval=scripts.evaluate:main",
        ]
    },
)
