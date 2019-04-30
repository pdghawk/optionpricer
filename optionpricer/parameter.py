""" Module for parameter classes

All classes should have at a minimum methods:
 - integral(time0,time1)
 - square_integral(time0,time1)
 - mean(time0,time1)

"""

class SimpleParam:
    def __init__(self,value):
        self.value = value

    def integral(time0,time1):
        return self.value*(time1-time0)

    def square_integral(time0,time1):
        return self.value**2*(time1-time0)

    def mean(time0,time1):
        return self.value
