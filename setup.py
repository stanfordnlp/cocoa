from setuptools import setup, find_packages
import sys

setup(name='cocoa',
      version='0.1',
      description='platform for dialogue research',
      packages=find_packages(exclude=('scraper', 'scripts', 'mutualfriends', 'negotiation', 'test')),
     )
