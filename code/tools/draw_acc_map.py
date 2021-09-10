import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif']=['Tahoma', 'DejaVu Sans',
                               'Lucida Grande', 'Verdana']
def plot_confusion_matrix(cm, classes,
                          savepath,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.5f' if normalize else 'd'

    thresh = cm.max() / 2.
    

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig(savepath)
    plt.close('all')

def draw_acc_loss_line(loss_list,acc_list,loss_pic_save_path,acc_pic_save_path,phase='train'):
    loss=np.array(loss_list)
    acc=np.array(acc_list)
    assert len(loss_list)==len(acc_list)
    epochs=range(1,len(loss_list)+1)
    plt.figure() 
    plt.plot(epochs, loss, 'r', label = '{} loss'.format(phase))
    plt.legend()
    plt.savefig(loss_pic_save_path)
    plt.figure() 
    plt.plot(epochs, acc, 'b', label = '{} acc'.format(phase))
    plt.legend()
    plt.savefig(acc_pic_save_path)
    plt.close('all')

if __name__ == '__main__':
    #cnf_matrix = np.ones([3,3])
    '''
    cnf_matrix=np.array([[1,2,3],[3,4,5],[4,5,14]])
    # cnf_matrix.shape == (3,3) in this example
    class_names = ['run', 'stand', 'walk']
    # Plot normalized confusion matrix

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix',
                          savepath='./test.jpg')
    '''
    draw_acc_loss_line([1,2,3],[5,7,9],'loss.pdf','acc.pdf',phase='train')
