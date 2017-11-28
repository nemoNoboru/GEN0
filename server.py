from socket import socket, AF_INET, SOCK_DGRAM
import json


class Gym:
    def __init__(self):
        self.serverSocket = socket(AF_INET, SOCK_DGRAM)
        self.serverSocket.bind(('', 3003))
        print "Server initialized"

    def sense(self):
        message, self.address = self.serverSocket.recvfrom(1024)
        print "sense"
        return json.loads(message)

    def step(self, action):
        message, self.address = self.serverSocket.recvfrom(1024)
        response = json.dumps({'action': action})
        self.serverSocket.sendto(response, self.address)
        return json.loads(message)
