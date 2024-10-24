import os
import unittest
import uuid

import pytest
from fs.errors import IllegalBackReference, CreateFailed
from fs.test import FSTestCases
from google.cloud.storage import Client

from fs_gcsfs import GCSFS

TEST_BUCKET = os.environ['TEST_BUCKET']


class TestGCSFS(FSTestCases, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = Client()
        cls.bucket = cls.client.get_bucket(TEST_BUCKET)
        super().setUpClass()

    def setUp(self):
        self.root_path = "gcsfs/" + str(uuid.uuid4())
        super().setUp()

    def tearDown(self):
        super().destroy_fs(self.fs)
        for blob in self.bucket.list_blobs(prefix=self.root_path):
            blob.delete()

    def make_fs(self):
        return GCSFS(bucket_name=TEST_BUCKET, root_path=self.root_path, client=self.client, create=True)


@pytest.fixture(scope="module")
def client_mock():
    class ClientMock:
        """A client mock class to instantiate GCSFS without making any requests in the constructor"""
        def get_bucket(self, _):
            pass
    return ClientMock()


@pytest.fixture(scope="module")
def client():
    return Client()


@pytest.fixture(scope="module")
def bucket(client):
    return client.get_bucket(TEST_BUCKET)


@pytest.fixture(scope="function")
def tmp_gcsfs(bucket, client):
    """Yield a temporary `GCSFS` at a unique 'root-blob' within the test bucket."""
    path = "gcsfs/" + str(uuid.uuid4())
    yield GCSFS(bucket_name=bucket.name, root_path=path, client=client, create=True)
    for blob in bucket.list_blobs(prefix=path):
        blob.delete()


@pytest.mark.parametrize("path,root_path,expected", [
    ("", None, ""),
    (".", None, ""),
    ("/", None, ""),
    ("foo", None, "foo"),
    ("/foo", None, "foo"),
    ("./foo", None, "foo"),
    ("foo/", None, "foo"),
    ("/foo/", None, "foo"),
    ("foo/bar", None, "foo/bar"),
    ("/foo/bar/", None, "foo/bar"),
    ("foo/../bar", None, "bar"),
    ("foo/../bar/..", None, ""),
    ("foo/../foo/bar", None, "foo/bar"),
    ("", "root_path", "root_path"),
    ("./", "root_path", "root_path"),
    ("foo", "root_path", "root_path/foo"),
    ("./foo", "root_path", "root_path/foo"),
    ("foo/../bar", "root_path", "root_path/bar"),
])
def test_path_to_key(path, root_path, expected, client_mock):
    gcs_fs = GCSFS(bucket_name=TEST_BUCKET, root_path=root_path, client=client_mock, strict=False)
    assert gcs_fs._path_to_key(path) == expected
    assert gcs_fs._path_to_dir_key(path) == expected + GCSFS.DELIMITER


def test_path_to_key_fails_if_path_is_parent_of_root_path(client_mock):
    gcs_fs = GCSFS(bucket_name=TEST_BUCKET, client=client_mock, strict=False)
    with pytest.raises(IllegalBackReference):
        gcs_fs._path_to_key("..")

    gcs_fs_with_root_path = GCSFS(bucket_name="bucket", root_path="root_path", client=client_mock, strict=False)
    with pytest.raises(IllegalBackReference):
        gcs_fs_with_root_path._path_to_key("..")


def test_listdir_works_on_bucket_as_root_directory(client):
    """Regression test for a bug fixed in 0.2.1"""
    gcs_fs = GCSFS(bucket_name=TEST_BUCKET, client=client, create=True)

    blob = str(uuid.uuid4())
    directory = str(uuid.uuid4())

    gcs_fs.touch(blob)
    gcs_fs.makedir(directory)

    result = gcs_fs.listdir("")

    # Manual clean-up of the created blobs
    gcs_fs.remove(blob)
    gcs_fs.removedir(directory)

    assert blob in result
    assert directory in result


@pytest.mark.parametrize("root_path", ["", ".", "/"])
def test_create_property_does_not_create_file_if_emptyish_root_path(root_path, client):
    """Regression test for a bug fixed in 0.4.1"""
    gcs_fs = GCSFS(bucket_name=TEST_BUCKET, root_path=root_path, client=client, create=True)
    assert gcs_fs.bucket.get_blob(root_path + GCSFS.DELIMITER) is None


def test_fix_storage_adds_binary_blobs_with_empty_string_as_directory_marker(bucket, tmp_gcsfs):
    # Creating a 'nested' hierarchy of blobs without directory marker
    for path in ["foo/test", "foo/bar/test", "foo/baz/test", "foo/bar/egg/test"]:
        key = tmp_gcsfs._path_to_key(path)
        blob = bucket.blob(key)
        blob.upload_from_string(b"Is this a test? It has to be. Otherwise I can't go on.")
    tmp_gcsfs.fix_storage()

    for path in ["", "foo", "foo/bar", "foo/baz", "foo/bar/egg"]:
        assert tmp_gcsfs.isdir(path)


def test_fix_storage_does_not_overwrite_existing_directory_markers_with_custom_content(bucket, tmp_gcsfs):
    for path in ["foo/test"]:
        key = tmp_gcsfs._path_to_key(path)
        blob = bucket.blob(key)
        blob.upload_from_string(b"Is this a test? It has to be. Otherwise I can't go on.")

    # Manual creation of 'directory marker' with custom content
    key = tmp_gcsfs._path_to_dir_key("foo/")
    blob = bucket.blob(key)
    content = b"CUSTOM_DIRECTORY_MARKER_CONTENT"
    blob.upload_from_string(content)

    tmp_gcsfs.fix_storage()

    assert blob.download_as_string() == content


def test_instantiation_fails_if_no_access_to_bucket():
    with pytest.raises(CreateFailed):
        GCSFS(bucket_name=str(uuid.uuid4()))


def test_instantiation_with_create_false_fails_for_non_existing_root_path():
    with pytest.raises(CreateFailed):
        GCSFS(bucket_name=TEST_BUCKET, root_path=str(uuid.uuid4()), create=False)