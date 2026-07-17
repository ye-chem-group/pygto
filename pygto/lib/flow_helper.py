import sys


LOG_ERROR = 1
LOG_WARN = 2
LOG_NOTE = 3
LOG_INFO = 4
LOG_DEBUG = 5


class StreamObject:
    ''' Base object providing logging and validated attribute updates.

        Attributes:
            verbose (int):
                Logging verbosity level. Default is `LOG_NOTE`.
            stdout (file-like object):
                Logging destination. Default is `sys.stdout`.
    '''
    verbose = 3
    stdout = sys.stdout

    def log_error(self, msg, verbose=None, space=True, indent=0):
        ''' Log an error-level message.

            Args:
                msg (str):
                    Message text.
                verbose (int):
                    Verbosity used for this message. Default is None, which uses
                    `self.verbose`.
                space (bool):
                    Whether to surround the message with blank lines. Default is True.
                indent (int):
                    Indentation level. Default is 0.
        '''
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_ERROR:
            self.write_msg(f'ERROR: {msg}', space, indent)

    def log_warn(self, msg, verbose=None, space=True, indent=0):
        ''' Log a warning-level message.

            Args:
                msg (str):
                    Message text.
                verbose (int):
                    Verbosity used for this message. Default is None, which uses
                    `self.verbose`.
                space (bool):
                    Whether to surround the message with blank lines. Default is True.
                indent (int):
                    Indentation level. Default is 0.
        '''
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_WARN:
            self.write_msg(f'WARN: {msg}', space, indent)

    def log_note(self, msg, verbose=None, space=False, indent=0):
        ''' Log a note-level message.

            Args:
                msg (str):
                    Message text.
                verbose (int):
                    Verbosity used for this message. Default is None, which uses
                    `self.verbose`.
                space (bool):
                    Whether to surround the message with blank lines. Default is False.
                indent (int):
                    Indentation level. Default is 0.
        '''
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_NOTE:
            self.write_msg(msg, space, indent)

    def log_info(self, msg, verbose=None, space=False, indent=0):
        ''' Log an information-level message.

            Args:
                msg (str):
                    Message text.
                verbose (int):
                    Verbosity used for this message. Default is None, which uses
                    `self.verbose`.
                space (bool):
                    Whether to surround the message with blank lines. Default is False.
                indent (int):
                    Indentation level. Default is 0.
        '''
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_INFO:
            self.write_msg(msg, space, indent)

    def log_debug(self, msg, verbose=None, space=False, indent=0):
        ''' Log a debug-level message.

            Args:
                msg (str):
                    Message text.
                verbose (int):
                    Verbosity used for this message. Default is None, which uses
                    `self.verbose`.
                space (bool):
                    Whether to surround the message with blank lines. Default is False.
                indent (int):
                    Indentation level. Default is 0.
        '''
        if verbose is None: verbose = self.verbose
        if verbose >= LOG_DEBUG:
            self.write_msg(msg, space, indent)

    def write_msg(self, msg, space=False, indent=0):
        ''' Write and flush a formatted message.

            Args:
                msg (str):
                    Message text.
                space (bool):
                    Whether to surround the message with blank lines. Default is False.
                indent (int):
                    Number of two-space indentation levels. Default is 0.
        '''
        if indent < 0:
            raise ValueError('negative indent')
        indent = '' if indent == 0 else '  ' * indent
        if space:
            self.stdout.write(f'\n{indent}{msg}\n' + '\n')
        else:
            self.stdout.write(f'{indent}{msg}' + '\n')
        self.flush_stdout()

    def flush_stdout(self):
        ''' Flush the logging destination when it supports flushing. '''
        if hasattr(self.stdout, 'flush'):
            self.stdout.flush()

    def set(self, **kwargs):
        ''' Set existing attributes and return the object.

            Args:
                kwargs (dict):
                    Attribute-value pairs. A function replacing a class method is
                    bound to this instance automatically.

            Return:
                self (StreamObject):
                    Modified object.
        '''
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
