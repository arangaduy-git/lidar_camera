#!/usr/bin/env python

"""
setup.py file for SWIG rplidar
"""

from setuptools import Extension, setup
from pathlib import Path


sourcefiles = ['rplidar_wrap.cxx', 'rplidar.cpp']

for path in Path('./rplidar_sdk/sdk/src/').glob('*.cpp'):
    sourcefiles.append(str(path))

for path in Path('./rplidar_sdk/sdk/src/arch/linux').glob('*.cpp'):
    sourcefiles.append(str(path))

for path in Path('./rplidar_sdk/sdk/src/hal').glob('*.cpp'):
    sourcefiles.append(str(path))  

for path in Path('./rplidar_sdk/sdk/src/dataunpacker').rglob('*.cpp'):
    sourcefiles.append(str(path))    

rplidar_module = Extension("_rplidar", sources=sourcefiles,
                            include_dirs=["./rplidar_sdk/sdk/include/", "./rplidar_sdk/sdk/src"],
                            language='c++')

setup (name = 'rplidar',
       version = '0.1',
       author      = "SWIG Docs",
       description = """Simple swig rplidar from docs""",
       ext_modules = [rplidar_module],
       py_modules = ["rplidar"],
       )