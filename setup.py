#!/usr/bin/env python3
"""
Setup configuration for ASCII Colorizer package.
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return [
            'Pillow>=9.0.0',
            'opencv-python>=4.0.0',
            'rich>=13.0.0',
            'colorama>=0.4.0',
            'numpy>=1.20.0'
        ]


def read_long_description():
    """Read the long description from README.md."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "A Python application that converts images and videos to colored ASCII art for terminal display."


setup(
    name='ascii-colorizer',
    version='1.0.0',
    author='ASCII Colorizer Team',
    author_email='ascii-colorizer@example.com',
    description='Convert images and videos to colored ASCII art',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ascii-colorizer/ascii-colorizer',
    
    packages=find_packages(),
    
    # Package data
    include_package_data=True,
    package_data={
        'ascii_colorizer': ['*.py'],
    },
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires='>=3.7',
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'ascii-colorizer=cli:main',
            'asciify=cli:main',  # Alternative shorter command
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Multimedia :: Video :: Conversion',
        'Topic :: Terminals',
        'Topic :: Artistic Software',
        'Environment :: Console',
    ],
    
    # Keywords for discoverability
    keywords=[
        'ascii', 'ascii-art', 'image-processing', 'video-processing',
        'terminal', 'ansi-colors', 'cli', 'converter', 'art', 'colorized'
    ],
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/ascii-colorizer/ascii-colorizer/issues',
        'Source': 'https://github.com/ascii-colorizer/ascii-colorizer',
        'Documentation': 'https://github.com/ascii-colorizer/ascii-colorizer/wiki',
    },
    
    # Additional options
    zip_safe=False,
    platforms=['any'],
    
    # Extras for optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.900',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
) 