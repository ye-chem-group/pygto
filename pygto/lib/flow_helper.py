import sys


LOG_ERROR = 1
LOG_WARN = 2
LOG_NOTE = 3
LOG_INFO = 4
LOG_DEBUG = 5


class StreamObject:
    ''' Similar to PySCF's `lib.StreamObject` with the following attributes:

        verbose : int
            Control log verbosity level. Default is 3 or LOG_NOTE.
        stdout :
            Destination for logging. Default is sys.stdout.
    '''
    verbose = 3
    stdout = sys.stdout

    def log_error(self, msg, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_ERROR:
            self.stdout.write(f'\nERROR: {msg}'+'\n\n')
            self.flush_stdout()

    def log_warn(self, msg, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_WARN:
            self.stdout.write(f'\nWARN: {msg}'+'\n\n')
            self.flush_stdout()

    def log_note(self, msg, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_NOTE:
            self.stdout.write(msg+'\n')
            self.flush_stdout()

    def log_info(self, msg, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_INFO:
            self.stdout.write(msg+'\n')
            self.flush_stdout()

    def log_debug(self, msg, verbose=None):
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_DEBUG:
            self.stdout.write(msg+'\n')
            self.flush_stdout()

    def flush_stdout(self):
        if hasattr(self.stdout, 'flush'):
            self.stdout.flush()

    def set(self, **kwargs):
        from types import MethodType
        import inspect

        sentinel = object()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                msg = f'{self.__class__.__name__} does not have attribute "{k}".'
                self.log_error(msg)
                raise AttributeError(msg)

            cls_attr = inspect.getattr_static(type(self), k, sentinel)
            if inspect.isfunction(cls_attr) and inspect.isfunction(v):
                setattr(self, k, MethodType(v, self))
            else:
                setattr(self, k, v)
        return self
