"""
Module Name: paths.py
Description: used to join absolute project root path with a relative path for easy I/O
Usage: from src.utils.paths import get_path 
"""

from pathlib import Path

#Go up to levels to get to the root of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_path(relative_path_str):

    return PROJECT_ROOT / relative_path_str