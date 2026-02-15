#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ForecastTRADE - Visualization Viewer
Displays model results and visualizations
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add paths
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    viewing_dir = os.path.join(os.path.dirname(__file__), 'viewing')
    sys.path.insert(0, src_dir)
    sys.path.insert(0, viewing_dir)
    
    # Change to viewing directory
    os.chdir(viewing_dir)
    
    # Execute view_results.py
    with open('view_results.py', 'r') as f:
        code = compile(f.read(), 'view_results.py', 'exec')
        exec(code)
