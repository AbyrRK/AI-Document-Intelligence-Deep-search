# frontend/server.py
import http.server
import socketserver
import os

PORT = 3000

# Serve from the frontend/ directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Frontend running at http://localhost:{PORT}")
    httpd.serve_forever()