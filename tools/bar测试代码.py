# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:58:53 2019

@author: HP
"""

import time
from progress.bar import Bar
if __name__=='__main__':
    import time
    x=100
    bar = Bar('Processing', max=100, fill='@')
    for i in range(100):
        time.sleep(0.1)
        x+=10
        bar.suffix='{ssss}'.format(ssss=x)
        bar.next()
    bar.finish()