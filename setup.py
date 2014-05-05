from setuptools import setup

setup(name='image-mining',
      version='0.1.4',
      author='Chris Adams',
      author_email='chris@improbable.org',
      packages=['image_mining'],
      scripts=['bin/extract-figures.py', 'bin/locate-thumbnail.py'],
      url='https://github.com/acdha/image-mining/',
      license='LICENSE.txt',
      description='Extract useful information from scanned images using OpenCV',
      long_description=open('README.rst').read(),
      install_requires=['numpy'])
