
from setuptools import setup, find_packages
  
setup( 
    name='Noise Robust Identifiability Physical Law Learning', 
    version='0.1.0', 
    description='A python package for identfying uniqueness in the physical law learning problem contaminated with noise', 
    author='Hillary Hauger', 
    author_email='hillary.hauger@yahoo.com', 
    packages=find_packages(), 
    install_requires=[ 
        'numpy', 
        'pandas', 
        'matplotlib',
        'seaborn',
        'pysindy==1.7.5',
        'sympy',
        'jupyterlab',
        'notebook',
    ], 
    python_requires='>=3.8',
) 