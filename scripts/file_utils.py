import os

def get_abs_file_path(file_path):
    """判断文件是否存在, 并返回file绝对路径"""
    file_path = os.path.abspath(file_path)
    file_path = file_path.replace("\\", "/")
    if not os.path.isfile(file_path):
        raise Exception("Error! {} is not exist or not a file!".format(file_path))
    return file_path

def get_abs_dir(dir):
    """判断目录是否存在, 并返回目录绝对路径"""
    dir = os.path.abspath(dir)
    dir = dir.replace("\\", "/")
    if not os.path.isdir(dir):
        raise Exception("Error! {} is not exist or not a dir!".format(dir))
    return dir

def create_dir(dir):
    """判断目录是否存在, 若不存在, 创建, 并返回目录绝对路径"""
    dir = os.path.abspath(dir)
    dir = dir.replace("\\", "/")
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def create_save_dir_from_file(file_path):
    """根据file路径判断file所在目录是否存在, 若不存在, 创建, 并返回file所在目录绝对路径"""
    file_path = os.path.abspath(file_path)
    file_path = file_path.replace("\\", "/")
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file_dir

def create_tmp_file_path(file_path):
    """根据file路径创建tmp file路径, 原则是在file文件名前缀后加_tmp, 比如：
       file_path为/home/abc/123.png, 那么创建文件名/home/abc/123_tmp.png, 
       这个函数的主要目的就是创建临时文件，不需要写很长的代码，直接调用该函数即可"""
    file_path = get_abs_file_path(file_path)
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    tmp_list = file_name.rsplit(".", 1)
    if len(tmp_list) == 1:
        tmp_file_path = file_path + "_tmp"
    else:
        tmp_file_path = os.path.join(dir_name, tmp_list[0]) + "_tmp." + tmp_list[1]
    return tmp_file_path

def main():
    tmp_file_path = create_tmp_file_path("/home/abc/123.png")
    print("tmp_file_path: ", tmp_file_path)

if __name__ == '__main__':
    main()