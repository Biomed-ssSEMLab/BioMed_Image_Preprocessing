'''logger.py - the Datadog logger'''

import collections
import datadog
import datetime
import os
import logging
import rh_logger
import rh_logger.api
import sys
import traceback

class DatadogLogger(rh_logger.api.Logger):
    '''Logger for datadog'''

    def __init__(self, name, config):
        self.name = name
        if "api-key" not in config:
            raise IndexError(
                "The api-key is missing from the datadog configuration "
                "subsection in the rh-logger section. See README.md for "
                "a configuration example."
            )
        api_key = config["api-key"]
        if name not in config:
            raise IndexError(
                ("The %s section is missing from the datadog configuration "
                 "subsection of the rh-logger section. See README.md for "
                 "a configuration example.") % name)
        if "app-key" not in config[name]:
            raise IndexError(
                 "There is no app-key in your application's logger "
                 "configuration section. See README.md for a configuration "
                 "example.")
        app_key = config[name]['app-key']
        datadog.initialize(api_key=api_key, app_key=app_key)

    def start_process(self, name, msg, args=None):
        '''Report the start of a process

        :param msg: an introductory message for the process
        '''
        if args is None:
            context = []
        elif isinstance(args, basestring):
            context = [ args ]
        elif isinstance(args, collections.Sequence):
            context = args
        else:
            context = [ str(args) ]
        datadog.api.Event.create(title="%s starting" % self.name,
                                 text=msg,
                                 alert_type="info",
                                 tags=[self.name, "startup"] + context)

    def end_process(self, msg, exit_code):
        '''Report the end of a process

        :param msg: an informative message about why the process ended
        :param exit_code: one of the :py:class: `ExitCode` enumerations
        '''
        if exit_code == rh_logger.ExitCode.success:
            datadog.api.Event.create(title="%s exiting" % self.name,
                                     text=msg,
                                     alert_type="success",
                                     tags=[self.name, "success"])
        else:
            datadog.api.Event.create(title="%s exiting with error" % self.name,
                                     text=msg,
                                     alert_type="error",
                                     tags=[self.name, "error", exit_code.name])

    def report_metric(self, name, metric, subcontext=None):
        '''Report a metric such as accuracy or execution time

        :param name: name of the metric, e.g. "Rand score"
        :param metric: the value
        :param subcontext: an optional sequence of objects identifying a
        subcontext for the metric such as a tile of the MFOV being processed.
        '''
        if isinstance(subcontext, collections.Sequence)\
           and not isinstance(subcontext, basestring):
            tags = [self.name] + subcontext
        elif subcontext is not None:
            tags = [self.name, subcontext]
        else:
            tags = [self.name]
        datadog.api.Metric.send(metric=name,
                                points=[metric],
                                host=self.name,
                                tags=tags)
        
    def report_metrics(self, name, time_series, context=None):
        if isinstance(context, collections.Sequence)\
           and not isinstance(context, basestring):
            tags = [self.name] + context
        elif context is not None:
            tags = [self.name, context]
        else:
            tags = [self.name]
        datadog.api.Metric.send(metric=name,
                                points=time_series.timestamps_and_metrics,
                                host=self.name,
                                tags=tags)

    def report_event(self, event, context=None, log_level=None):
        '''Report an event

        :param event: the name of the event, for instance, "Frobbing complete"
        :param context: a subcontext such as "MFOV: 5, Tile: 3"
        '''
        if isinstance(context, collections.Sequence)\
           and not isinstance(context, basestring):
            tags = [self.name] + context
        else:
            tags = [self.name, context]
        if log_level is None or log_level in (logging.DEBUG, logging.INFO):
            alert_type="info"
        elif log_level == logging.WARNING:
            alert_type="warning"
        elif log_level in (logging.ERROR, logging.CRITICAL):
            alert_type="error"
        else:
            alert_type="info"
            
        datadog.api.Event.create(title=event,
                                 text=event,
                                 alert_type=alert_type,
                                 tags=tags)

    def report_exception(self, exception=None, msg=None):
        '''Report an exception

        :param exception: the :py:class: `Exception` that was thrown. Default
        is the one reported by sys.exc_info()
        :param msg: an informative message
        '''
        if exception is None:
            exc_type, exception, tb = sys.exc_info()
        else:
            exc_type = type(exception)
            tb = None
        if msg is None:
            msg = str(exception)
        tags = [self.name, "exception", exc_type.__name__]
        if tb is not None:
            # TODO: add stack breadcrumbs to the tags
            #       Consider using Sentry for logging exceptions
            msg += "\n" + "".join(traceback.format_exception(
                exc_type, exception, tb))
        datadog.api.Event.create(title="Exception report",
                                 text=msg,
                                 alert_type="error",
                                 tags=tags)
        datadog.api.Metric.send(metric="exception",
                                points=[(datetime.datetime.now(),
                                         1)],
                                type="counter",
                                host=self.name,
                                tags=tags)


def get_logger(name, config):
    return DatadogLogger(name, config)
