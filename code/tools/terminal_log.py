import logging

if __name__!='__main__':
    from . import godblessdbg
import os
def create_log_file_terminal(logfile_path,log_name='log_name'):#.txt
    '''
    todo:create a log object
    input : logfile_path (I often use the txt file)
            log_name str %(name)-12s
    return:a log object
    use obj.info('str') you can output the str into the log file and terminal at the same time
    '''

    log=logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    # set a format which is simpler for console use
    console_formatter = logging.Formatter('%(asctime)s-%(name)-12s: terminal %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(console_formatter)
    log.addHandler(console)
    
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   
    file=logging.FileHandler(logfile_path) 
    file.setFormatter(file_formatter)
    log.addHandler(file)
    
#    print(godblessdbg.godbless)
    log.info(godblessdbg.begin)

    return log

import shutil

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if os.path.exists(os.path.join(path, 'scripts')):
            shutil.rmtree(os.path.join(path, 'scripts'))
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))

            if os.path.isdir(script):
                shutil.copytree(script, dst_file)
            else:
                shutil.copyfile(script, dst_file)
            
            

def save_opt(opt,log_path):

    if '.txt' in log_path:
        txt_save_path=log_path
    else:
        txt_save_path=os.path.join(log_path,'opt_save.txt')

    if not os.path.exists(os.path.dirname(txt_save_path)):
        os.makedirs(os.path.dirname(txt_save_path))

    
    with open(txt_save_path,'w') as f:
        for k, v in opt.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(opt, k))
                f.write('{0} \t {1} \n'.format(k, getattr(opt, k)))


if __name__=='__main__':
    import godblessdbg
    log_txt=create_log_file_terminal('./b.txt')
    log_txt.info('aa')
