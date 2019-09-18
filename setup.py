# -*- coding: utf-8 -*-
"""
Setup script.
Created on Wed Sep 18 10:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/rcnnpose-pytorch

"""


from setuptools import setup, find_packages


setup(name='rcnnpose-pytorch',
      version='0.1.0',
      description='Mask R-CNN and Keypoint R-CNN wrapper for pose estimation with PyTorch',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Prasun Roy',
      author_email='prasunroy.pr@gmail.com',
      url='https://github.com/prasunroy/rcnnpose-pytorch',
      license='MIT',
      install_requires=[
              'numpy',
              'opencv-contrib-python'
      ],
      classifiers=[
              'Development Status :: 4 - Beta',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.7',
              'Topic :: Scientific/Engineering',
              'Topic :: Software Development :: Libraries',
              'Topic :: Software Development :: Libraries :: Python Modules',
              'Topic :: Utilities'
      ],
      keywords=[
              'rcnnpose',
              'pytorch',
              'mask-rcnn',
              'keypoint-rcnn',
              'pose-estimation',
              'keypoint-estimation',
              'computer-vision',
              'machine-learning'
      ],
      packages=find_packages())
