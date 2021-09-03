import torch

def save_model(net,acc,epoch,optimizer,loss,global_step,lr,save_path):
    state = {
            'net': net.state_dict() ,
            'acc': acc,
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'global_step':global_step,
            'lr':lr,
            }
    torch.save(state, save_path)

def load_model(net,load_path,optimizer=None):
    checkpoint = torch.load(load_path)
    net.load_state_dict(checkpoint['net'])
    save_epoch=checkpoint['epoch']
    save_acc=checkpoint['acc']
    save_global_step=checkpoint['global_step']
    save_lr=checkpoint['lr']

    if optimizer==None:
        return net,save_epoch,save_acc,save_global_step,save_lr
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return net,save_epoch,save_acc,save_global_step,save_lr,optimizer