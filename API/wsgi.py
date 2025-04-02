import sys
import os

# Ensure the parent directory is in the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    app.run()