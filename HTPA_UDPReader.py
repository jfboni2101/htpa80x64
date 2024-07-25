# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

from queue import Queue
import time

from readers import HTPA_UDPReader
from tparray import TPArray
from threads import UDP, Imshow

# %% Parameters provided by the user
# Resolution of sensor
width = 80
height = 64

# Broadcasting has not been implemented yet, therefore the IP of the HTPA
# device needs to be known in advance
UDP_IP = '192.168.1.167'

# %% Code begins here

tparray = TPArray()
udp_buffer = Queue(maxsize=1)

# Create instance of UDP reader
udp_reader = HTPA_UDPReader(width, height)

# Create instance of UDP thread
udp_thread = UDP(udp_reader=udp_reader,
                 dev_ip=UDP_IP,
                 write_buffer=udp_buffer)

# Create instance of thread plotting the reveived data
plot_thread = Imshow(width=width,
                     height=height,
                     read_buffer=udp_buffer)

if __name__ == '__main__':

    udp_thread.start()
    plot_thread.start()

    for i in range(1):
        time.sleep(120)
        udp_thread.stop()
        plot_thread.stop()



