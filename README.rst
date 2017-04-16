Contours Processor
===========

Python module to build Dicoms (inputs) and Contours (targets) dataset genertor.

There are currently two main capabilities of this package.

1. **ContourFileExtractor**:  Extract Contours and its associated Dicoms as Numpy Arrays into a target folder.

2. **ContourDataset**: Generate Batches of inputs and outputs to feed into a deep learning model


Python
------
Tested in Python 3.5, but should be compatible with Python 2.7


Installation
=============

.. code-block:: bash

    $ pip install git+git://github.com/sampathweb/contours-processor.git@master


Usage
======

.. code-block:: python

    >>> from contours_processor import ContourFileExtractor, ContourDataset

- `Notebook Examples and Usage <https://github.com/sampathweb/contours-processor/tree/master/example-usage.ipynb>`_


Tests
=====

.. code-block:: python

    >>> pytest

Currently only tests creation of objects.  Additional tests need to be written before using it.


Limitations
=========

Usage and Limitations are documents in the Examples notebook <https://github.com/sampathweb/contours-processor/tree/master/example-usage.ipynb>`_