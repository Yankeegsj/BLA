import numpy as np
import torch.nn as nn
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def get_one_arch(
                 num_pool_layers=5, 
                 num_pool_operations=2,
                 num_other_operations=2, 
                 num_search_layer=12):
    pool_op=np.random.randint(0,num_pool_operations,num_pool_layers)
    other_op=np.random.randint(num_pool_operations,num_pool_operations+num_other_operations,num_search_layer-num_pool_layers)
    arch=np.concatenate([pool_op,other_op])
    np.random.shuffle(arch)
    return arch.tolist()#list


def generate_archs(n=100,
                 num_pool_layers=5, 
                 num_pool_operations=2,
                 num_other_operations=2, 
                 num_search_layer=12):
    archs=[]
    archs=[get_one_arch(num_pool_layers,num_pool_operations,num_other_operations,num_search_layer) for i in range(n)]
    return archs

def arch_to_seq(arch):
    seq=list(map(lambda x:x+1,arch))
    return seq
def seq_to_arch(seq):
    archs=list(map(lambda x: x-1, seq))
    return archs

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


class earlystop():
    def __init__(self, 
                 min_delta=0,
                 patience=5):
        self.best_loss=[1e15]
        self.patience=patience
        self.min_delta=min_delta
        self.wait=0
        self.stopped_flag = False
        self.start_flag=False

    def check(self,current_loss):
        if len(current_loss)>1 and not self.start_flag:
            self.best_loss=[1e15 for i in range(len(current_loss))]
            self.start_flag=True
        
        judge=((np.array(self.best_loss)-np.array(current_loss))-self.min_delta)>0

        if judge.any():
            self.best_loss = np.where(judge,np.array(current_loss),np.array(self.best_loss)).tolist()
            self.wait=0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_flag = True
            

class lr_decay():
    def __init__(self, 
                 lr_decay_factor,
                 min_lr,
                 min_delta=0,
                 patience=5
                 ):
        self.best_loss=[1e15]
        self.lr_decay_factor=lr_decay_factor
        self.patience=patience
        self.min_delta=min_delta
        self.min_lr=min_lr
        self.wait=0
        self.decay_flag = False
        self.start_flag=False

    def check(self,current_loss,current_lr):

        if len(current_loss)>1 and not self.start_flag:
            self.best_loss=[1e15 for i in range(len(current_loss))]
            self.start_flag=True

        judge=((np.array(self.best_loss)-np.array(current_loss))-self.min_delta)>0

        if judge.any():
            self.best_loss = np.where(judge,np.array(current_loss),np.array(self.best_loss)).tolist()
            self.wait=0
        else:
            self.wait += 1
            if self.wait >= self.patience and current_lr*self.lr_decay_factor>=self.min_lr:
                self.decay_flag = True
                self.wait=0



class AverageMeter_array(object):
    """Computes and stores the average and current value"""

    def __init__(self,save_len):
        self.save_len=save_len
        self.reset()

    def reset(self):
        self.val = np.zeros([self.save_len])
        self.avg = np.zeros([self.save_len])
        self.sum = np.zeros([self.save_len])
        self.count = np.zeros([self.save_len])

    def update(self, val):
        if len(val.shape)==2:
            n,l=val.shape
        elif len(val.shape)==1:
            val=val[np.newaxis,:]
            n,l=val.shape
        assert l==self.save_len
        self.val = val
        for i in range(self.save_len):
            self.sum[i]+= np.sum(val[:,i])
            self.count[i] += n
        self.avg = self.sum / self.count

class AverageMeter_acc_map(object):
    """Computes and stores the average and current value"""

    def __init__(self,save_len):
        self.save_len=save_len
        self.reset()

    def reset(self):
        self.avg = np.zeros([self.save_len])
        self.sum = np.zeros([self.save_len])
        self.count = np.zeros([self.save_len])

    def update(self, pred,gt):
        # pred , gt  b*save_len*h*w  np.array
        if len(pred.shape)==4:
            batch=pred.shape[0]
            for i in range(self.save_len):
                if np.sum(gt[:,i,:])>0:
                    self.sum[i]+=batch*np.sum(np.logical_and(pred[:,i,:],gt[:,i,:]))/np.sum(gt[:,i,:])*100
                    
            self.count+=batch
            self.avg = self.sum / self.count
        elif len(pred.shape)==3:
            #b*h*w
            batch=pred.shape[0]
            for i in range(self.save_len):
                if np.sum(gt[:,i,:])>0:
                    self.sum[i]+=batch*np.sum(np.logical_and(pred==i,gt[:,i,:]))/np.sum(gt[:,i,:])*100
            self.count+=batch
            self.avg = self.sum / self.count



class AverageMeter_array_mask(object):
    """Computes and stores the average and current value"""

    def __init__(self,save_len):
        self.save_len=save_len
        self.reset()

    def reset(self):
        self.val = np.zeros([self.save_len])
        self.avg = np.zeros([self.save_len])
        self.sum = np.zeros([self.save_len])
        self.count = np.zeros([self.save_len])

    def update(self, val,mask):
        assert val.shape==mask.shape
        if len(val.shape)==2:
            n,l=val.shape
        elif len(val.shape)==1:
            val=val[np.newaxis,:]
            mask=mask[np.newaxis,:]
            n,l=val.shape
        assert l==self.save_len
        self.val = val
        for i in range(n):
            for j in range(self.save_len):
                if mask[i,j]:
                    self.sum[j]+= val[i,j]
                    self.count[j] += 1
                    self.avg[j]=self.sum[j] / self.count[j]




if __name__ == '__main__':
    archs=generate_archs(n=1000,
                 num_pool_layers=5, 
                 num_pool_operations=2,
                 num_other_operations=2, 
                 num_search_layer=12)#长度为n的list list中每个都是list
    encoder_input_=list(map(lambda x:arch_to_seq(x),archs))
    arch_=list(map(lambda x:seq_to_arch(x),encoder_input_))
    print(encoder_input_[0])
    print(arch_[0])