import setuptools

setuptools.setup(
    description="Image reading layer for the Rhoana pipeline",
#     install_requires=[
#         "pyaml>=15.8.2"
#         ],
    name="rh_img_access_layer",
    packages=["rh_img_access_layer"],
    dependency_links = ['http://github.com/adisuissa/gcsfs/tarball/master#egg=fs_gcsfs-0.4.1'],
    url="https://github.com/Rhoana/rh_img_access_layer",
    version="0.0.1"
)
