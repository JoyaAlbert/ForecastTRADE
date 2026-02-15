#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ForecastTRADE - Main Entry Point
Runs the LSTM-XGBoost hybrid model
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add src and viewing to Python path BEFORE changing directory
    src_dir = os.path.join(project_root, 'src')
    viewing_dir = os.path.join(project_root, 'viewing')
    
    sys.path.insert(0, src_dir)
    sys.path.insert(0, viewing_dir)
    sys.path.insert(0, project_root)
    
    # Change to src directory for relative imports to work
    os.chdir(src_dir)
    
    # Execute main.py
    with open('main.py', 'r') as f:
        code = compile(f.read(), 'main.py', 'exec')
        exec(code)
