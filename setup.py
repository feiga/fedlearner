from setuptools import setup, find_packages

setup(
    name='fedlearner',
    version='0.1',
    packages=find_packages(),
    data_files=[('cc', ['cc/embedding.so'])],
    include_package_data=True,
    install_requires=[
        'click'
    ],
    entry_points='''
        [console_scripts]
        fedlearner=fedlearner.cli.cli:cli_group
    ''',
)
