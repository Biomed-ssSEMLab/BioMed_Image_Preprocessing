import setuptools

setuptools.setup(
    description="Logging and metric reporting for Rhoana pipeline",
    entry_points={
        "rh_logger.backend": [
            "default = rh_logger.backends.backend_python_logging:get_logger",
            "datadog = rh_logger.backends.backend_datadog_logging:get_logger",
            "carbon = rh_logger.backends.backend_carbon_logging:get_logger"
        ]
    },
    install_requires=["enum34>=1.0.4",
                      "datadog>=0.11.0"],
    dependency_links = ['http://github.com/Rhoana/rh_config/tarball/master#egg=rh_config-0.1.0'],
    name="rh_logger",
    packages=["rh_logger", "rh_logger.backends"],
    url="https://github.com/Rhoana/rh_logger",
    version="2.0.1"
)
