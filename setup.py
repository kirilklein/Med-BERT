#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(name='medbert',
      version='1.0',
      description='Pytorch Implementation of MedBERT',
      long_description=readme,
      author='Kiril Klein',
      author_email='kikl@di.ku.dk',
      url="https://github.com/kirilklein/Med-BERT.git",
      packages=['medbert'],
     )