"""Start the EMPOWER Explorer:  python EMPOWER_Explorer.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from empower_explorer.app import main

if __name__ == '__main__':
    main()
