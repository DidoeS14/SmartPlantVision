import logging
import inspect

class ContextFilter(logging.Filter):
    """Injects caller info (filename, class, func) into log records."""

    def filter(self, record):
        # Walk the stack to find caller outside logging module and this file
        frame = inspect.currentframe()
        while frame:
            info = inspect.getframeinfo(frame)
            # Skip frames from logging module and this logger file itself
            if info.filename != __file__ and "logging" not in info.filename:
                break
            frame = frame.f_back

        if frame:
            # Extract filename, function, class from the frame
            record.caller_file = info.filename.split('/')[-1]  # filename only
            record.caller_line = info.lineno
            record.caller_func = info.function

            # Try to get class name if exists
            cls = None
            if 'self' in frame.f_locals:
                cls = type(frame.f_locals['self']).__name__
            elif 'cls' in frame.f_locals:
                cls = frame.f_locals['cls'].__name__
            record.caller_class = cls if cls else '-'
        else:
            record.caller_file = record.caller_line = record.caller_func = record.caller_class = '-'

        return True


def setup_logger(name="SmartGV", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s '
            '[%(caller_file)s:%(caller_class)s:%(caller_func)s:%(caller_line)d] '
            '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ch.setFormatter(formatter)

        # Add filter to inject caller context
        ch.addFilter(ContextFilter())

        logger.addHandler(ch)

    return logger
