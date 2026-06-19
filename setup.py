#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='scmtg',
      version='0.0.1',
      packages=find_packages(),
      description='scMTG learns generative Markov transitions for single-cell temporal dynamics',
      long_description='',

      author='Xuejian Cui',
      author_email='cuixj@hit.edu.cn',
      url="https://github.com/liuq-lab/scMTG",
      python_requires='>=3.8.0',
      license='MIT',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.8',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     
    install_requires=[
        'tensorflow-gpu==2.6.0',
        'keras==2.6.0',
    ]
     )