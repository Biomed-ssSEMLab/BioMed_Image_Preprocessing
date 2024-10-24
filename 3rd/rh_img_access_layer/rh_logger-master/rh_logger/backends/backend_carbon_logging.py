'''backend_carbon_logging.py

This logger logs metrics to Carbon / Graphite. Event and other logging is
to the Python logger.

Configuration:

rh-logger:
    carbon:
        host: 127.0.0.1
        port: 2003
'''

import rh_logger
import backend_python_logging
import Queue
import socket
import threading
import time

class CarbonLogger(backend_python_logging.BLPLogger):
    '''Log metrics to Carbon/Graphite'''

    def __init__(self, name, config):
        super(CarbonLogger, self).__init__(name)
        if "host" not in config:
            host = "127.0.0.1"
        else:
            host = config["host"]
        if "port" not in config:
            port = 2003
        else:
            port = config["port"]
        s = socket.socket()
        s.connect((host, port))
        self.queue = Queue.Queue()
        self.thread = threading.Thread(
            target = lambda :self.run(s),
            name = "CarbonLoggerThread")
        self.thread.start()
    
    def run(self, s):
        '''Thread for running the logger'''
        
        done = False
        while not done:
            msg = self.queue.get()
            if msg is None:
                break
            while True:
                try:
                    another_msg = self.queue.get_nowait()
                    if another_msg is None:
                        done = True
                        break
                    msg += another_msg
                except Queue.Empty:
                    break
            s.sendall(msg)
        s.close()
    
    def make_name(self, name):
        name = name.replace(" ", "_")
        return "%s.%s" % (self.name, name)
    
    def report_metrics(self, name, time_series, context=None):
        '''Report a series of metrics'''
        full_name = self.make_name(name)
        msg = "".join(["%s %f %f\n" % (full_name, v, t) 
                       for t, v in time_series.timestamps_and_metrics])
        self.queue.put(msg)
    
    def report_metric(self, name, metric, subcontext=None):
        full_name = self.make_name(name)
        self.queue.put("%s %f %f\n" % (full_name, metric, time.time()))
    
    def end_process(self, msg, exit_code):
        super(CarbonLogger, self).end_process(msg, exit_code)
        self.queue.put(None)


def get_logger(name, config):
    return CarbonLogger(name, config)
