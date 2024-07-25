# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:08:22 2024

@author: rehmer
"""

import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from tparray import TPArray



class HTPA_ByteStream_Converter():
    def __init__(self, width, height, **kwargs):

        self.width = width
        self.height = height

        self.tparray = TPArray()

        # Initialize an array to which to write data
        self.data_cols = self.tparray.get_serial_data_order()
        self.data = np.zeros((len(self.data_cols)))

        self.output = kwargs.pop('output', 'np')

        # Depending on the specified output format, the convert method is
        # pointing to _bytes_to_np() or _bytes_to_pd()
        if self.output == 'np':
            self.convert = self._bytes_to_np
        elif self.output == 'pd':
            self.convert = self._bytes_to_pd

    def _bytes_to_np(self, byte_stream: list):

        if not isinstance(byte_stream, list):
            TypeError('byte_stream needs to be a list of bytes')

        # Zero data array
        self.data.fill(0)

        j = 0

        # Loop over all elements / packages in list
        for package in byte_stream:

            # Read the first byte, it's the package index, and throw away
            _ = package[0]

            # Loop over all bytes and combine MSB and LSB
            idx = np.arange(1, len(package), 2)

            for i in idx:
                self.data[j] = int.from_bytes(package[i:i + 2], byteorder='little')
                j = j + 1

        Warning("Check if pixels are in appropriate order (i.e. if Bodo's)" + \
                "and pyplots/opencvs coordinate system are the same")

        return self.data

    def _bytes_to_pd(self, byte_stream: list):

        self._bytes_to_np(byte_stream)

        df_data = pd.DataFrame(data=[self.data],
                               columns=self.data_cols)

        return df_data

    # def bytes_to_img(self,byte_stream):

    #     # Loop over all bytes and combine MSB and LSB
    #     idx = np.arange(0,len(byte_stream),2)

    #     img = np.zeros((1,self.width*self.height))
    #     j=0

    #     for i in idx:
    #         img[0,j] = int.from_bytes(byte_stream[i:i+2], byteorder='little')
    #         j = j+1

    #     img = img.reshape((self.height,self.width))
    #     img = np.flip(img,axis=1)

    #     return img


class HTPA_UDPReader():

    def __init__(self, width, height, **kwargs):

        self._output = None
        self.width = width
        self.height = height

        # Port over which HTPA device connects, usually 30444
        self._port = kwargs.pop('port', 30444)

        # Initialize a TPArray object, which contains all information regarding
        # how data is stored, organized and transmitted for this array type
        self._tparray = TPArray()

        # Create a DataFrame to store all bound devices in
        self.col_dict = {'IP': str, 'MAC-ID': str, 'Arraytype': int,
                         'status': str}
        self.index = pd.Index(data=[], dtype=int, name='DevID')
        self._devices = pd.DataFrame(data=[],
                                     columns=self.col_dict.keys(),
                                     index=self.index)

        # Dictionary for storing all sockets in
        self._sockets = {}

        # Initialize ByteStream Reader for convertings bytes to pandas
        # dataframes
        self.bytestream_converter = HTPA_ByteStream_Converter(width, height, **kwargs)

        # depending on the desired output type, choose which method of
        # HTPA_ByteStream_Reader should be used to parse bytes to that
        # type

        if self.output == 'np':
            self.bytes_to_output = self.bytestream_reader.bytes_to_np
        # if self.output == 'pd':
        #     self.bytes_to_output = self.bytestream_reader.bytes_to_pd

    @property
    def output(self):
        return self._output

    @property
    def port(self):
        return self._port

    @property
    def tparray(self):
        return self._tparray

    @property
    def devices(self):
        return self._devices

    @devices.setter
    def devices(self, df):
        devices_old = self._devices
        self._devices = pd.concat(devices_old, df)

    @property
    def sockets(self):
        return self._sockets

    @sockets.setter
    def sockets(self, socket: dict):
        self._sockets.update(socket)

    def broadcast(self):
        Warning('Not implemented yet.')
        return None

    def _read_port(self, udp_socket, server_address):
        """
        Read all packages available at the specified port and return the last
        one

        Parameters
        ----------
        server_address : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        clear = False

        packages = []

        # In order to prevent and endless loop that reads in package after
        # package (e.g. if Appset is still continuously streaming), read in
        # a maximum amount of packages that equals 10 whole frames
        p_max = 50 * self.tparray._package_num
        p = 0

        while clear == False:

            try:
                package = udp_socket.recv(self.tparray._package_size)
                packages.append(package)
                p = p + 1
            except:
                clear = True
                break

            if p > p_max:
                clear = True
                break

            time.sleep(10 / 1000)

        return packages

    def bind_tparray(self, ip: str):
        """
        Creates a socket for the device with the given ip

        Parameters
        ----------
        ip : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Create the udp socket
        udp_socket = socket.socket(socket.AF_INET,  # Internet
                                   socket.SOCK_DGRAM)  # UDP

        # Create server address
        server_address = (ip, self._port)

        # Set timeout to 1 second
        # socket.setdefaulttimeout(1)
        udp_socket.settimeout(1)

        # Try calling the device and check if it's a HTPA device

        # Stop any stream that might still continue, e.g. if program
        # crashed

        # Send message to device to stop streaming bytes
        for i in range(5):
            udp_socket.sendto(bytes('X', 'utf-8'), server_address)
            time.sleep(10 / 1000)

        # Clear the port in case old packages are still on it
        _ = self._read_port(udp_socket, server_address)

        # Try to call device
        _ = udp_socket.sendto(bytes('Calling HTPA series devices',
                                    'utf-8'),
                              server_address)

        # The package following the call should contain device information
        call = self._read_port(udp_socket, server_address)

        # If calling was successfull, extract basic information from the
        # answer string

        call_fail = False

        if len(call) == 1:
            try:
                dev_info = self._callstring_to_information(call[0])
            except:
                call_fail = True
        else:
            call_fail = True

        if call_fail == True:
            Exception('Calling HTPA series device failed')
            return None

        # Next try to bind the device that answered the call
        try:
            _ = udp_socket.sendto(bytes('Bind HTPA series device',
                                        'utf-8'),
                                  server_address)

            # Read the answer to the bind command from socket
            _ = self._read_port(udp_socket, server_address)
        except:
            Exception('Calling HTPA series device failed')
            return None

        dev_info['status'] = 'bound'

        # Add socket to dictionary
        dev_id = dev_info.index.item()
        self.sockets = {dev_id: udp_socket}

        # Append new device to device list and store
        self._devices = dev_info

        print('Bound HTPA device with DevID: ' + str(dev_id))

        return self.devices.copy()

    def release_tparray(self, dev_id):

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to stop streaming bytes
        for i in range(5):
            udp_socket.sendto(bytes('X', 'utf-8'), server_address)
            time.sleep(10 / 1000)

        # Clean up port
        answ = self._read_port(udp_socket, server_address)

        # Send message to release device
        _ = udp_socket.sendto(bytes('x Release HTPA series device', 'utf-8'),
                              server_address)

        # Clean up port
        answ = self._read_port(udp_socket, server_address)

        print('Released HTPA device with DevID: ' + str(dev_id))

        return answ

    def start_continuous_bytestream(self, dev_id):

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('K', 'utf-8'), server_address)

    def stop_continuous_bytestream(self, dev_id):

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to stop streaming bytes
        for i in range(5):
            udp_socket.sendto(bytes('X', 'utf-8'), server_address)
            time.sleep(10 / 1000)

        # Clean up port
        answ = self._read_port(udp_socket, server_address)

        return answ

    def read_continuous_bytestream(self, dev_id):

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one device has the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create variable that indicates if DataFrame was constructed success-
        # fully
        success = False
        while success == False:

            # Initialize list for received packages
            packages = []

            # Read incoming packages until one with the package index 1 is received
            sync = False

            while sync == False:

                # Receive package
                try:
                    package = udp_socket.recv(self.tparray._package_size)
                except socket.timeout:
                    return np.zeros((len(self.tparray._serial_data_order),))

                # Check package index
                # Exception for (32,32)
                if self.tparray._npsize == (32, 32):

                    if len(package) == 1292:
                        package_index = 1
                    else:
                        package_index = 2
                else:
                    # All other arrays
                    package_index = int(package[0])

                # Check if it is equal to 1
                if package_index == 1:
                    packages.append(package)
                    sync = True

            for p in range(2, self.tparray._package_num + 1):

                # Receive package
                try:
                    package = udp_socket.recv(self.tparray._package_size)
                except socket.timeout:
                    return np.zeros((len(self.tparray._serial_data_order),))

                # Get index of package
                # Exception for (32,32)
                if self.tparray._npsize == (32, 32):
                    if len(package) == 1288:
                        package_index = 2
                    else:
                        package_index = 0
                else:
                    # All other arrays
                    package_index = int(package[0])

                # Check if package index has the expected value
                if package_index == p:
                    packages.append(package)

            # In the end check if as many packages were received as expected
            if len(packages) == self.tparray._package_num:

                # If yes, pass the packages to the class that parses the
                # bytes into an np.ndarray or pd.DataFrame
                # df_frame = self.bytestream_reader.bytes_to_df(packages)

                frame = self.bytestream_converter.convert(packages)

                success = True

            else:
                print('Frame lost.')
                sync = False

        return frame

    # def read_single_frame(self,dev_id,**kwargs):

    #     # Read voltage 'c' or temperature 'k' frame
    #     mode = kwargs.pop('mode','k')

    #     # Get device information from the device list
    #     dev_info = self.devices.loc[[dev_id]]

    #     # If more than one devices have the same device id, return error
    #     if len(dev_info) != 1:
    #         Exception('Multiple devices have the same device id.')

    #     # Get udp socket
    #     udp_socket = self.sockets[dev_id]

    #     # Create server address, i.e. tuple of device IP and port
    #     server_address = (dev_info['IP'].item(), self.port)

    #     # Send message to device to send the single frame
    #     mess = udp_socket.sendto(bytes(mode,'utf-8'),server_address)

    #     # Michaels code is atrocious. Sometimes it sends no frame, sometimes
    #     # 1 frame and sometimes two frames. All these cases have to be considered
    #     # Try to get first frame (works always)
    #     try:
    #         frame = self._receive_udp_frame(udp_socket)
    #     except:
    #         frame = pd.DataFrame(data=[],
    #                              columns = self.tparray.get_serial_data_order())

    #     if frame is None:
    #         frame = pd.DataFrame(data=[],
    #                              columns = self.tparray.get_serial_data_order())

    #     if len(frame) == 0:
    #         print('Reading frame failed.')
    #     else:
    #         print('Successfully received frame')

    #     # Clean up whole port in which a second frame may or may not be
    #     self._read_port(udp_socket)

    #     return frame

    def _receive_udp_frame(self, udp_socket):
        """
        Receives UDP packages and tries to put them together to a frame

        Parameters
        ----------
        udp_socket : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # Initialize list for received packages
        packages = []

        # Read incoming packages until one with the package index 1 is received
        sync = False

        while sync == False:

            # Receive package
            package = udp_socket.recv(self.tparray._package_size)

            # Get index of package
            package_index = int(package[0])

            # Check if it is equal to 1
            if package_index == 1:
                packages.append(package)
                sync = True

        for p in range(2, self.tparray._package_num + 1):

            # Receive package
            package = udp_socket.recv(self.tparray._package_size)

            # Get index of package
            package_index = int(package[0])

            # Check if package index has the expected value
            if package_index == p:
                packages.append(package)

        # In the end check if as many packages were received as expected
        if len(packages) == self.tparray._package_num:

            frame = self.bytestream_converter.convert(packages)

        else:
            print('Frame lost.')
            frame = None

        return frame

    def device_put_to_sleep(self, dev_id, **kwargs):
        """
        Puts a device to sleep
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('s', 'utf-8'), server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')

        if answer.strip() == 'Module set into sleep mode.':
            print('Device ' + str(dev_id) + ' put to sleep.')
            return True
        else:
            print('Putting device ' + str(dev_id) + ' to sleep failed.')
            return False

    def device_wake_up(self, dev_id, **kwargs):
        """
        Wakes device up
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('S', 'utf-8'), server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')

        if answer.strip() == 'Module woke up.':
            print('Device ' + str(dev_id) + ' awake.')
            return True
        else:
            print('Failed waking up device ' + str(dev_id) + '.')
            return False

    # def _device_is_streaming(self,udp_socket,server_address):
    #     """
    #     Check if the device is already streaming

    #     Parameters
    #     ----------
    #     udp_socket : TYPE
    #         DESCRIPTION.
    #     server_address : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """

    #     # Try to read in packages and check if they have the size expected from
    #     # a HTPA device

    #     # Try to read in a maximum amount of packages that corresponds to
    #     # 10 frames
    #     p_max = 10 * self.tparray._package_num

    #     for p in range(p_max):

    #         try:
    #             package = udp_socket.recv(self.tparray._package_size)

    #     for i in range()

    def _callstring_to_information(self, call: bytes):
        """
        Extracts information from the ridiculously long and inefficient answer
        to the device call.

        Parameters
        ----------
        call : bytes
            DESCRIPTION.

        Returns
        -------
        dev_info : TYPE
            DESCRIPTION.

        """

        call = call.decode('utf-8')

        # Extract information from callstring
        arraytype = int(call.split('Arraytype')[1].split('MODTYPE')[0])
        mac_id = re.findall(r'\w{2}\.\w{2}\.\w{2}\.\w{2}.\w{2}.\w{2}', call)[0]
        ip = re.findall(r'\d{3}\.\d{3}\.\d{3}.\d{3}', call)[0]

        # Remove leading zeros from IP
        ip = '.'.join([str(int(x)) for x in ip.split('.')])

        dev_id = int(call.split('DevID:')[1].split('Emission')[0])

        dev_info = pd.DataFrame(data=[],
                                columns=self.col_dict.keys(),
                                index=self.index)

        dev_info.loc[dev_id, ['IP', 'MAC-ID', 'Arraytype']] = \
            [ip, mac_id, arraytype]

        dev_info = dev_info.astype(self.col_dict)

        return dev_info