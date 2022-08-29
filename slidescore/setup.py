from setuptools import setup, find_packages
import os

os.chdir('../')

setup(name='slidescore', version='0.1', 
      packages=['.slidescore', '.image']
      )

