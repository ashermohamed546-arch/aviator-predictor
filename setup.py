from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='aviator-predictor',
    version='0.1.0',
    author='Asher Mohamed',
    author_email='ashermohamed546@example.com',
    description='Machine learning-powered prediction system for Aviator betting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ashermohamed546-arch/aviator-predictor',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'tensorflow>=2.8.0',
        'keras>=2.8.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'isort>=5.9.0',
        ],
    },
)
