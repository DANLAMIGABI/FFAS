#http://www.huyng.com/posts/python-performance-analysis/

import time

class Timer(object):
    def __init__(self, tag='', verbose=False):
        self.verbose = verbose
        self.tag = tag #string for indicating what was timed

    def __enter__(self):
        self.start = time.time()
        return self

    def getTime(self): #get time in seconds
        self.now = time.time()
        self.secs = self.now-self.start
        return round(self.secs, 2)
    
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = round(self.end - self.start,5)
        self.msecs = round(self.secs * 1000, 2) # millisecs
        if self.verbose:
            print tag, ":  ",
            print 'elapsed time: %f ms' % self.msecs