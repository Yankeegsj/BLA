# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:18:14 2019

@author: HP
"""
'''
2019.8月添加:read_csv(csv_file) 忽略编码错误读取csv,读取csv直接用这个就行
        return reader
        #使用时直接for line in reader:即可

2019.8.9添加Logger类的参数recreate=True
    默认如果存在csv新写入而是重新开一个文件,仅在做数据集时不能继续写入时才令recreate=True
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
        #打开路径,没有会自己创建
        #这样可以直接继续添加信息
        #默认会重新写入
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
        #写入行,不断写入行记录数据
        if not flag or recreate:
            self.logger.writerow(header)
        #保存标题行
        self.header = header

    def __del(self):
        #文件关闭后重新写入会删除之前的数据
        #所以只需要关闭文件,下次write时会删除所有数据
        self.log_file.close()
      
    #写入运行日志 
    def log(self, values):
        
        #value传入的是dict，需要加上这段索引为header的元素
        ###################
        write_values = []
        for col in self.header:
            #col是否再values中存在索引，存在才能存入对应的索引下
            #不存在会报错
            assert col in values
            write_values.append(values[col])
        ####################
        #传入列表就不需要上面那段且加上这句  write_values=values
        
        self.logger.writerow(write_values)
        #flush将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入
        self.log_file.flush()

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def read_csv(csv_file):
    f=open(csv_file,encoding='utf-8',errors='ignore')
    reader=csv.reader(f)
    #使用时直接for line in reader:即可
    return reader




def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    
    #求出每行的最大值所在位置传给pred即对应标签
    #沿着给定维度返回K个最值largest=True返回最大值
    #dim=1,每行求k个最值，返回列向量
    #dim=0,每行求k个最值，返回行向量
    #torch.topk(k, dim=None, largest=True, sorted=True)
    _, pred = outputs.topk(1, 1, True)
    #转置
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
