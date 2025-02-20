from setuptools import setup, find_packages


setup(
    name="liger",
    version="0.0.1",
    description="Helper functions for the Likert General Regressor project",
    author="Chris Reeves",
    author_email="reeves.chris.allan@gmail.com",
    url="https://github.com/reevesc7/liger",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "dill==0.3.9",
        "matplotlib==3.10.0",
        "numpy==2.2.2",
        "openai==1.61.1",
        "pandas==2.2.3",
        "scikit_learn==1.6.1",
        "sentence_transformers==3.4.1",
        "git+https://github.com/reevesc7/tpot.git#egg=tpot",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12.8',            # Specify the Python versions you support
)
