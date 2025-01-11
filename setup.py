from setuptools import setup

setup(
    name='pyMethTools',
    version='0.2.0',    
    description='A Python package for analysis of targetted methylation data',
    url='https://github.com/AndyCGraham/methSimpy',
    author='Andy Graham',
    author_email='andygraham7162@gmail.com',
    license='BSD 2-clause',
    packages=['pyMethTools'],
    install_requires=['numpy',
                      'scipy', 
                      'pandas',
                      'ray',
                      'matplotlib',
                      'seaborn'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3.11',
    ],
)
