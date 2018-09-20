# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:46:44 2018

@author: Bara
"""
import time
import socket
import struct
import sys
import select

class TCPIP(object):

    def __init__(self, local_ip, local_port, remote_ip, remote_port, debug = False):
        # server information for server_lisent
        self.local_ip = local_ip
        self.local_port = local_port
        self.isClientConnect = False
        self.clientAddress = []
        self.connection = []
        # client information for bind
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        # enable or disaple extra information for debugging purpocess
        self.debug = debug

    # @init_bind_connection: establish binding on remote ip and port 
    def init_bind_connection(self):
        print('Starting a bind connection on ip:{!r}'.format(self.remote_ip),
              '  port:{!r}'.format(self.remote_port))
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.bind((self.remote_ip, self.remote_port))
        print (self.sock.getsockname())
        #self.sock.settimeout(100)

    # @ init_server: establish aserver of local ip and port       
    def init_server(self): 
        self.server_address = (self.local_ip, self.local_port)
        print('Starting up a server on:{} port:{}'.format(*self.server_address))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(self.server_address)
    
    # @server_listen: wait for any client to connect with the server
    def server_listen(self):
        # Wait for a connection
        self.sock.listen(1)
        if self.debug: print('waiting for a connection')
        self.connection, self.clientAddress = self.sock.accept()
        if self.debug: print('accept connection from', self.clientAddress)
        self.isClientConnect = True
        self.sock.settimeout(0.002)
        return self.isClientConnect
    
    # @recieve: get number of double variables sent from client or remote IP
    def recieve(self, bufSize, indata = None):
        dataNum = []
        dt = []
        try:
            # get current time stamp
            timeStamp = time.time()
            # wait for data from remote client
            if self.isClientConnect:
                dataRaw = self.connection.recv(bufSize)
                address = self.clientAddress
            else:
                dataRaw, address = self.sock.recvfrom(bufSize)
            # display info
            if self.debug:
                print('received from {!r}'.format(address),
                      'this raw message:', dataRaw)
            # if non-emputy message
            if dataRaw:
                # find number of sent numbers
                numOfValues = int(len(dataRaw) / 8)
                # decode message to 'BigEndian' double numbers
                dataStr = struct.unpack('>' + 'd' * numOfValues, dataRaw)
                # store numbers
                dataNum = []
                for i in range(0, numOfValues):
                    dataNum.append(float(dataStr[i]))
                
                # calculate time difference
                dt = time.time() - timeStamp
                # display result
                if self.debug:
                    print ('number of recieved data', numOfValues)
                    print('time difference = ', dt, ' data :', dataNum)
                    print('data :', dataNum)
                
                if not indata == None:
                    sentData = struct.pack('!d' * 1, indata)
                    #if self.debug:
                    print(indata, sentData)
                    self.connection.send(sentData)
            # if no message
            else:
                print('no data from', address)
            
            return dataNum, dt
        
        except:
            print('error in server_get function')
            print ("Unexpected error:", sys.exc_info()[0])
            self.close()
            raise
            
    #@empty_socket: remove the data present on the socket"""
    def empty_socket(self):    
        while 1:
            print("select")
            inputready, o, e = select.select([self.sock],[],[], 0.0)
            print(inputready)
            if len(inputready)==0:
                break
            for s in inputready:
                print("read empty_socket")
                self.recieve(1024)

    def FlushListen(self):
        for i in range(3):
            print("FlushListen")
            try:
                self.recieve(1024)
            except:
                break;
                
    # @close: close socket connection
    def close(self):
        print('Close soket')
        self.sock.close()

if __name__ == "__main__":
    LOCAL_IP = "localhost"
    LOCAL_PORT = 18001
    REMOTE_IP = "localhost"
    REMOTE_PORT = 18001
    tcpip = TCPIP(LOCAL_IP,LOCAL_PORT,REMOTE_IP,REMOTE_PORT, False)
    #tcpip.init_bind_connection();
    tcpip.init_server()
    tcpip.server_listen()
    i = 0
    volt = 0
    t0 = time.time();
    for i in range(100):
        data, dt = tcpip.recieve(1024, volt)
        t = time.time() - t0
        print ("data = ", data)
    tcpip.close()

