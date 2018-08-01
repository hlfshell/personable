from setuptools import setup

setup(name='personable',
      version='0.1',
      description='Frame by frame person recognition and tracking',
      url='https://github.com/hlfshell/personable',
      author='Keith Chester',
      author_email='kchester@gmail.com',
      license='MIT',
      packages=['personable'],
      install_requires=[
          'face-recognition>=1.22',
          'face-recognition-models>=0.3.0',
          'opencv-python>=3.4.1.15'
      ],
      zip_safe=False)