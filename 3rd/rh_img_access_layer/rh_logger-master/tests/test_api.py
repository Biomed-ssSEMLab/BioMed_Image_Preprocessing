'''Test the logging API'''

from rh_logger import logger
import rh_logger
import unittest


class TestLogging(unittest.TestCase):

    def test_backend(self):
        #
        # Try all of the logging functions to get enough coverage to
        # run through the code
        #
        logger.start_process("foo", "Hello, world", ["bar", "baz"])
        logger.report_metric("Execution time", "14 Mahayugas")
        logger.report_event("Frobbing the galactopus", "very messy this time")
        try:
            raise Exception("Whoops")
        except:
            logger.report_exception()
        logger.end_process("bye for now", rh_logger.ExitCode.success)

    def test_get_set_logger_backend(self):
        old_backend = rh_logger.get_logging_backend()
        try:
            rh_logger.set_logging_backend("foo")
            self.assertEqual(rh_logger.get_logging_backend(), "foo")
        finally:
            rh_logger.set_logging_backend(old_backend)
