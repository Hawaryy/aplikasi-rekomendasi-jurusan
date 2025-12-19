import sys
import os

# Tambahkan path aplikasi
sys.path.insert(0, os.path.dirname(__file__))

# Import Flask app dari app.py
from app import app as application

# Untuk compatibility dengan berbagai WSGI server
if __name__ == "__main__":
    application.run()