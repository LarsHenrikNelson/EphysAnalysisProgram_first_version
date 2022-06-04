# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:15:48 2021

@author: LarsNelson
"""
import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_deps = ['pyqtgraph',
         'matplotlib',
         'pyqt5',
         'bottleneck',
         'qdarkstyle',
         'scipy',
         'statsmodels',
         'bottleneck',
         'numpy',
         'xlsxwriter'
        ]

setuptools.setup(
    name="EphysAnalysisProgram",
    author="Lars Nelson",
    version="0.0.1"
    author_email="lhn6@pitt.edu",
    description="Program to analyze patch clamp ephys data",
    install_requires=install_deps,
    classifiers=[
        "Development Status :: 3 - Alpha"
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
        "Licencse :: OSI Approved :: GNU General Public License vs (GPLv3)",
        "Operating System :: OS Indpendent",
        ],
    keywords='electrophysiology',
    packages=find_packages(where='src'),
    python_requires='>=3.8')