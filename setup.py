from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="LQTMomentMag",
    version="1.0.0",
    author= "Arham Zakki Edelo",
    author_email= "edelo.arham@gmail.com",
    description= "Calculate seismic moment magnitude in full LQT energy for very local earthquake case",
    long_description= long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/bgjx/LQTMomentMag",
    license="MIT",
    keywords='Seismology, Moment Magnitude, Spectral Fitting, LQT Component',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "scipy>=1.9.0",
        "obspy>=1.4.0",
        "tqdm>=4.64.0",
        "configparser>=5.2.0",
    ],
    entry_points={
        "console_scripts": [
            "LQTMwcalc = LQTMomentMag.main:main",
        ]
    },
    python_requires = ">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)