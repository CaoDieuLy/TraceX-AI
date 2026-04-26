from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Health check endpoints (covers all common paths)
        if self.path in ['/health', '/', '/api/health', '/status', '/ping']:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "message": "server is running"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            print(f"✅ GET {self.path} - 200")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"error": "not found", "path": self.path}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            print(f"❌ GET {self.path} - 404")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            request_data = json.loads(post_data.decode('utf-8'))
            response_message = f"received: {request_data.get('data', 'no data')}"
        except json.JSONDecodeError:
            response_message = "invalid json"

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {"message": response_message}
        self.wfile.write(json.dumps(response).encode('utf-8'))
        print(f"✅ POST {self.path} - 200")

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f'starting httpd server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
