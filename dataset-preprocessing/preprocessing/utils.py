# -*- coding: utf-8 -*-

__author__ = "Ariel Rodrigues"
__version__ = "0.1.0"
__license__ = ""

"""
Module Docstring
"""

from preprocessing import tools as dbpedia_tools


def define_toolset(type):
    if type == 'dbpedia':
        return dbpedia_tools
