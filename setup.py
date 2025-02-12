from setuptools import setup, find_packages

setup(
    name='ml_functions',
    version='0.1.0',
    description='Funções de Machine Learning para Predição de Resultados de Futebol',
    author='Felipe',
    author_email='felipe@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.1',
        'pandas>=2.1.1',
        'scikit-learn>=1.6.1',
        'matplotlib>=3.7.2',
        'seaborn>=0.13.2'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
) 