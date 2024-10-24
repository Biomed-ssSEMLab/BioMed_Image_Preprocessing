# rh_logger
## A package for logging / process status / metrics reporting for rhoana

How to use:

* `from rh_logger import logger``
* `logger.start_process(process_name, msg, args)`: log the start of a process.
The args should contain enough information to distinguish this instantiation
from another that produces substantially different data.
* `logger.end_process(msg)`: log the end of a process
* `logger.report_metric(name, metric, subcontext=None)`: Report a metric such as accuracy or execution time. Subcontext gives enough information to narrow the metric to an instance of the named step.
* `logger.report_event(event, context=None)`: Report an event.  Context gives enough information to narrow the metric to an instance of the named step.
* `logger.report_exception(exception=None, msg=None)`: Report an exception. Exception is last exception thrown by default. Message is `str(exception)` by default.
* After calling `get_logger`, you can reference the logger globally as
rh_logger.logger
* If you write a logging backend, the configuration for it is stored in
`rh_logger.logging_config`

## Configuring:

rh_logger uses entry points to register and find its backend. The entry point
group is "rh_logger.backend". rh_logger comes with a default logger that logs
via Python logging and the datadog logger. You choose the logger either
programatically, specifying the entry point name in a call to
`rh_logger.set_logging_backend` or specify the name in your .rh-config.yaml
file:

### rh_config entries

    rh-logger:
        logging-backend: default or datadog (or your own)
        default:
            # values here are passed into logging.config.dictConfig
            # for example
            version: 1
            handlers:
              console:
                class : logging.StreamHandler
                formatter: default
                level: INFO
            formatters:
              brief:
                format: '%(message)s'
              default:
                format: '%(asctime)s %(levelname)-8s %(name)-15s %(message)s'
                datefmt: '%Y-%m-%d %H:%M:%S'
            root:
              handlers: [console]
              level: INFO
        datadog:
            api-key: 0000000000000000000
            my-first-application:
                app-key: 000000000000000000000
            my-second-application:
                app-key: 000000000000000000000
        carbon:
            host: 127.0.0.1
            port: 2003
            version: 1
            handlers:
                console:
                    class : logging.StreamHandler
                    formatter: default
                    level: INFO
            formatters:
                brief:
                    format: '%(message)s'
                default:
                    format: '%(asctime)s %(levelname)-8s %(name)-15s %(message)s'
                    datefmt: '%Y-%m-%d %H:%M:%S'
            root:
                handlers: [console]
                level: INFO


## Datadog logger

Datadog is a centralized console and API for monitoring a distributed
application (https://app.datadoghq.com). You can use rh_logging with the
datadog logger if you have a datadog account and have API and APP keys
for your application.

To configure:
* Set the logging-backend rh_config entry to `datadog`
* Set the api-key config entry to your datadog API key
* Create an entry under `datadog` for your app, using the name that you
will use in the call to `get_logger`. Specify the app-key as sub-entry
* Use `rh_logger.get_logger()` to get yourself a logger

## Carbon logger

Carbon and Graphite (and grafana) are open source applications for logging
metrics data. The Carbon logger logs metrics to Carbon, but uses the Python
logger for events and similar. This means that the config for the Carbon
logger includes both the host and port # for the Carbon server and the
config parameters for the Python logger.
