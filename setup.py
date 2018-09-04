from setuptools import setup

setup(
    name='vggish',
    packages=['vggish'],
    include_package_data=True,
    install_requires=[
        'keras',
        'sklearn',
        'numpy',
        'h5py',
        'resampy'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)