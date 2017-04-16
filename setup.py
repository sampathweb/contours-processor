#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    # TODO: Add package version requirements here
    "pydicom",
    "pillow",
    "numpy"
]

setup(
    name='contours-processor',
    version='0.0.1',
    description="Load Contours and Dicom Files",
    long_description=readme,
    author="Ramesh Sampath",
    author_email='.',
    url='http://github.com/sampathweb/contours-processor',
    packages=[
        'contours_processor',
    ],
    package_dir={'contours_processor':
                 'contours_processor'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='contours_processor',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ]
)
