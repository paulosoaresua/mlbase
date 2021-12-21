from setuptools import setup, find_packages


with open('requirements.txt') as f:
    libs = [lib.strip() for lib in f.readlines() if lib]

    setup(
        name='MLBase',
        version='0.1.0_dev',
        description='General structure to evaluate ML models.',
        author='Paulo Soares',
        packages=find_packages(),
        install_requires=libs,
        author_email='paulosoares@email.arizona.edu'
    )
