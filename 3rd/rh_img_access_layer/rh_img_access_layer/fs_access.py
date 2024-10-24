import os
from fs import open_fs
from urllib.parse import urlparse
import google.auth.exceptions
import time



class FSAccessRegistry(object):
    """
    A per-process (Singleton) filesystem access layer, that holds all the relevant file system access objects
    """

    MAX_TRIES = 3

    def __new__(cls):
        if not hasattr(cls, '__instance'):
            cls.__instance = super(FSAccessRegistry, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self._registered_fs = {}

    @staticmethod
    def _normalize_url(url):
        if "://" not in url and ":/" in url:
            url = url.replace(":/", "://")
        return url

    def _open(self, url, rw_str):
        url = FSAccessRegistry._normalize_url(url)
        parsed_url = urlparse(url)
        if parsed_url.scheme == "file":
            url_unique_id = "osfs:///"
        else:
            url_unique_id = "{}://{}".format(parsed_url.scheme, parsed_url.netloc)
        if url_unique_id not in self._registered_fs:
            fs = open_fs(url_unique_id)
            self._registered_fs[url_unique_id] = fs
        else:
            fs = self._registered_fs[url_unique_id]
        try_counter = 0
        while try_counter <= FSAccessRegistry.MAX_TRIES:
            try:
                res = fs.open(parsed_url.path, rw_str)
                return res
            except:
                try_counter += 1
                if try_counter > FSAccessRegistry.MAX_TRIES:
                    raise
                print("Re-accessing path: {} ({})".format(url, rw_str))
        # should not reach here
        return None


    def open_read(self, url, binary=True):
        read_str = "rb" if binary else "rt"
        return self._open(url, read_str)

    def open_write(self, url, binary=True):
        write_str = "wb" if binary else "wt"
        return self._open(url, write_str)

    def exists(self, url):
        url = FSAccessRegistry._normalize_url(url)
        parsed_url = urlparse(url)
        if parsed_url.scheme == "file":
            url_unique_id = "osfs:///"
        else:
            url_unique_id = "{}://{}".format(parsed_url.scheme, parsed_url.netloc)
        if url_unique_id not in self._registered_fs:
            fs = open_fs(url_unique_id)
            self._registered_fs[url_unique_id] = fs
        else:
            fs = self._registered_fs[url_unique_id]
        try_counter = 0
        while try_counter <= FSAccessRegistry.MAX_TRIES:
            try:
                # Check if it's a file/concrete object
                res = fs.exists(parsed_url.path)
                # TODO - check if needed: If not, check if it is a directory
                # if not res:
                #     res = 
                return res
            except:
                try_counter += 1
                if try_counter > FSAccessRegistry.MAX_TRIES:
                    raise
                print("Re-accessing path: {} ({})".format(url, rw_str))
        # should not reach here
        return None

    def isdir(self, url):
        url = FSAccessRegistry._normalize_url(url)
        parsed_url = urlparse(url)
        if parsed_url.scheme == "file":
            url_unique_id = "osfs:///"
        else:
            url_unique_id = "{}://{}".format(parsed_url.scheme, parsed_url.netloc)
        if url_unique_id not in self._registered_fs:
            fs = open_fs(url_unique_id)
            self._registered_fs[url_unique_id] = fs
        else:
            fs = self._registered_fs[url_unique_id]
        try_counter = 0
        while try_counter <= FSAccessRegistry.MAX_TRIES:
            try:
                # Check if it's a file/concrete object
                res = fs.isdir(parsed_url.path)
                # TODO - check if needed: If not, check if it is a directory
                # if not res:
                #     res = 
                return res
            except:
                try_counter += 1
                if try_counter > FSAccessRegistry.MAX_TRIES:
                    raise
                print("Re-accessing path: {} ({})".format(url, rw_str))
        # should not reach here
        return None

def attempt(func):
    MAX_ATTEMPTS = 3
    def wrapper(*args):
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            try:
                return func(*args)
            except google.auth.exceptions.RefreshError as e:
                attempt += 1
                time.sleep(attempt)
                if attempt == MAX_ATTEMPTS:
                    raise e
    return wrapper



class FSAccess(object):

    def __init__(self, _url, binary, read=True):
        self._url = _url
        self._binary = binary
        self._read = read

    @attempt
    def __enter__(self):
        if self._read:
            self._handle = FSAccessRegistry().open_read(self._url, self._binary)
        else:
            self._handle = FSAccessRegistry().open_write(self._url, self._binary)
        return self._handle

    def __exit__(self, type, value, traceback):
        return self._handle.__exit__(type, value, traceback)

    @staticmethod
    @attempt
    def exists(path):
        return FSAccessRegistry().exists(path)

    @staticmethod
    @attempt
    def isdir(path):
        return FSAccessRegistry().isdir(path)

    @staticmethod
    @attempt
    def isdir(path):
        return FSAccessRegistry().isdir(path)

    @staticmethod
    def get_fs_parsed_url(url):
        parsed_url = urlparse(url)
        url_prefix = "{}://{}".format(parsed_url.scheme, parsed_url.netloc)
        return url_prefix, parsed_url.path

    @staticmethod
    @attempt
    def fs_create_dir(output_dir):
        output_dir = FSAccessRegistry._normalize_url(output_dir)

        fs_loc, fs_path = FSAccess.get_fs_parsed_url(output_dir)
        with open_fs(fs_loc) as out_fs:
            if not out_fs.exists(fs_path):
                out_fs.makedirs(fs_path)


