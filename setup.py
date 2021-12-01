from setuptools import setup

setup(
    name='dragonfly',
    version='0.0.1',
    entry_points = {
        'console_scripts': ['dragonfly=dragonfly.src.core.main:main'],
    }
)
