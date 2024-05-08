from setuptools import find_packages, setup

with open("sbm/README.md", "r", encoding="utf8") as f:
    long_description = f.read()

setup(
    name="space-breakdown-method",
    version="0.0.2",
    description="Space Breakdown Method (SBM) is a clustering algorithm developed for Spike Sorting handling overlapping and imbalanced data. Improved Space Breakdown Method (ISBM) is the updated and improved version of SBM. A new algorithm for the detection of brain oscillations packets has been developed based on SBM, called Time-Frequency Breakdown Method (TFBM)",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/ArdeleanRichard/Space-Breakdown-Method",
    author="Eugen-Richard Ardelean",
    author_email="ardeleaneugenrichard@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scikit-learn", "matplotlib", "scipy", "networkx", "plotly", "pandas"],
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.7",
)
