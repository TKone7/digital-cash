import socketserver, socket 
import sys 

host = "0.0.0.0"
port = 10000
address = (host, port)

class MyTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class TCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        message = self.request.recv(10).strip()
        print(f"got a message {message}")
        
        if message == b"ping":
            self.request.sendall(b"pong\n")
def serve():
    server = MyTCPServer(address, TCPHandler)
    server.serve_forever();

def ping():
    # create instance of a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    sock.sendall(b"ping")
    data = sock.recv(10).decode()
    print(f"Received {data}")

if __name__ == "__main__":
    #parse command line args
    command = sys.argv[1]
    if command == "ping":
        ping()
    elif command == "serve":
        serve()
    else:
        print("not a valid command")
