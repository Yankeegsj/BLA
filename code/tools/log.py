# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:18:14 2019

@author: HP
"""
'''

'''
import csv
import os
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    
    def __init__(self, path, header,recreate=False):

        if os.path.exists(path):
            flag=True
            if recreate:
                self.log_file = open(path, 'w',newline='')
            else:
                self.log_file = open(path, 'a',newline='')
        else:
            flag=False
            self.log_file = open(path, 'a',newline='')
        
        self.logger = csv.writer(self.log_file, dialect=("excel"))

        if not flag or recreate:
            self.logger.writerow(header)

        self.header = header

    def __del(self):

        self.log_file.close()

    def log(self, values):
        

        write_values = []
        for col in self.header:

            assert col in values

        self.logger.writerow(write_values)

        self.log_file.flush()

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def read_csv(csv_file):
    f=open(csv_file,encoding='utf-8',errors='ignore')
    reader=csv.reader(f)

    return reader




def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    

    _, pred = outputs.topk(1, 1, True)

    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size
        
if __name__=='__main__':
    for i in range(2):
        log=Logger('./test.csv',['header1','header2'])
        log.log({
                'header1':'h1',
                'header2':'h2',
                })
