# -*- coding: utf-8 -*-

""" Created on 9:43 AM, 9/4/19
    @author: ngunhuconchocon
    @brief: Пролетарии всех стран, соединяйтесь! да здравствует наша советская родина
"""

from setuptools import setup

setup(
    name="ZaG2P",
    version="0.0.1",
    author="ngunhuconchocon",
    description="Convert non-Vietnamese word to Vietnamese phonemes/syllables",
    # license="BSD",
    url="https://github.com/enamoria/ZaG2P",
    install_requires=[
          'python-Levenshtein', 'torch>=1.1.0', 'torchtext==0.3.1', 'dill', 'visdom', 'PyYAML'
      ],
    packages=['ZaG2P'],
    include_package_data=True
)
