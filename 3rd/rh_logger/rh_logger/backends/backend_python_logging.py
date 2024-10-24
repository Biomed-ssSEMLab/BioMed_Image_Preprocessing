'''backend_python_logging.py - logging backend using Python logging

Note: this is intended only for informal debugging.
'''

import logging
import logging.config
import rh_logger
import rh_logger.api
import sys


class BLPLogger(rh_logger.api.Logger):

    def __init__(self, name, config):
        if "version" not in config or config["version"] != 1:
            # If no config supplied, use basic config
            logging.basicConfig()
            logging.root.setLevel(logging.INFO)
        else:
            logging.config.dictConfig(config)
        self.logger = logging.getLogger(name)

    def start_process(self, name, msg, args=None):
        '''Report the start of a process

        :param msg: an introductory message for the process
        '''
        if args is not None:
            self.logger.info("Starting process: %s (%s)" %
                             (msg, repr(args)))
        else:
            self.logger.info("Starting process: %s")

    def end_process(self, msg, exit_code):
        '''Report the end of a process

        :param msg: an informative message about why the process ended
        :param exit_code: one of the :py:class: `ExitCode` enumerations
        '''
        self.logger.info("Ending process: %s, exit code = %s" %
                         (msg, exit_code.name))

    def report_metric(self, name, metric, subcontext=None):
        '''Report a metric such as accuracy or execution time

        :param name: name of the metric, e.g. "Rand score"
        :param metric: the value
        :param subcontext: an optional sequence of objects identifying a
        subcontext for the metric such as a tile of the MFOV being processed.
        '''
        if subcontext is None:
            self.logger.info("Metric %s=%s" %
                             (name, str(metric)))
        else:
            self.logger.info("Metric %s=%s (%s)" %
                             (name, str(metric), subcontext))

    def report_metrics(self, name, time_series, context=None):
        times = [_[0] for _ in time_series.timestamps_and_metrics]
        metrics = [_[1] for _ in time_series.timestamps_and_metrics]
        delta = times[-1] - times[0]
        total = sum(metrics)
        avg = float(total) / len(metrics)
        msg = "Metric %s: Running time = %0.4f, avg = %f, total = %f" % (
            name, delta, avg, total)
        if context is None:
            self.logger.info(msg)
        else:
            self.logger.info(msg + " (%s)" % str(context))

    def report_event(self, event, context=None, log_level=None):
        '''Report an event

        :param event: the name of the event, for instance, "Frobbing complete"
        :param context: a subcontext such as "MFOV: 5, Tile: 3"
        '''
        if log_level is None:
            log_fn = self.logger.info
        elif log_level <= logging.DEBUG:
            log_fn = self.logger.debug
        elif log_level <= logging.INFO:
            log_fn = self.logger.info
        elif log_level <= logging.WARNING:
            log_fn = self.logger.warning
        elif log_level <= logging.ERROR:
            log_fn = self.logger.error
        elif log_level <= logging.CRITICAL:
            log_fn = self.logger.critical
        else:
            log_fn = self.logger.critical
        if context is None:
            log_fn(event)
        else:
            log_fn("%s (%s)" % (event, repr(context)))

    def report_exception(self, exception=None, msg=None):
        '''Report an exception

        :param exception: the :py:class: `Exception` that was thrown. Default
        is the one reported by sys.exc_info()
        :param msg: an informative message
        '''
        if exception is None:
            if msg is None:
                msg = str(sys.exc_value)
            self.logger.exception(msg, exc_info=1)
        else:
            if msg is None:
                msg = str(exception)
            self.logger.error(msg)


def get_logger(name, config):
    '''Get the default rh_logging logger'''
    return BLPLogger(name, config)
