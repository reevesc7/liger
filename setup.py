from setuptools import setup, find_packages


setup(
    name="liger",
    version="0.2.1",
    entry_points={ "console_scripts": [
        "liger-pipeline=liger.run_tpot_pipeline:main",
    ]},
    description="Helper functions for the Likert General Regressor project",
    author="Chris Reeves",
    author_email="reeves.chris.allan@gmail.com",
    url="https://github.com/reevesc7/liger",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    install_requires=[
        "dill>=0.3.9,<0.4.0",
        "matplotlib>=3.10.0,<4.0.0",
        "numpy>=1.26.4,<3.0.0",
        "openai>=1.61.1,<2.0.0",
        "pandas>=2.2.3,<3.0.0",
        "scikit_learn>=1.4.2,<2.0.0",
        "sentence_transformers>=3.4.1,<4.0.0",
        "tpot>=1.0.0,<2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8, <3.11',            # Specify the Python versions you support
)
