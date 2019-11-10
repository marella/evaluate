from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

name = 'evaluate'

setup(
    name=name,
    version='0.0.0',
    description=long_description.splitlines()[0],
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ravindra Marella',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'pandas',
        'xgboost',
        'lightgbm',
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='{} scikit-learn sklearn xgboost lightgbm machine-learning'.
    format(name),
)
