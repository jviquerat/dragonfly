from setuptools import setup

setup(
    name='dgf',
    version='0.0.1',
    install_requires=['gym'],
    entry_points = {
        'console_scripts': ['dgf=dragonfly.src.core.main:main']
    }
)
