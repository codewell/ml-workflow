from setuptools import setup, find_packages


setup(
    name='ml-workflow',
    version='0.2.0',
    description='Workflow tools for pytorch and ignite',
    author='Felix Abrahamsson, Richard Löwenström, Jim Holmström',
    author_email='richard@aiwizo.com',
    keywords='pytorch ignite workflow utilities',
    packages=find_packages(exclude=['venv', 'dist', 'recipes']),
    install_requires=[
        'torch>=1.3.1',
        'ignite>=0.3.0',
        'tqdm>=4.41.1',
        'opencv-python>=4.1.2.30',
        'pandas>=0.25.3',
    ],
)
