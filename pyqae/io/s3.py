"""Tools for making interacting with object store and buckets easier, particularly since most tools assume you are in the US"""
import os
import tempfile
from urllib.parse import urlparse


_test_url = 's3://ls-output/20170927/TNM_StagingD20170411T141157857101_by_ThomasW_of_ACC23381854/.run_success'
_test_region = 'eu-central-1'
_test_bucket = 'ls-annotations'


def urljoin(*args):
    """
    Joins given arguments into a url. Trailing but not leading slashes are
    stripped for each argument. (urllib.parse.urljoin does not work for the following test cases (which is what we need))
    >>> urljoin('s3:/', 'bob', 'is_a_file', 'all.npz')
    's3://bob/is_a_file/all.npz'
    >>> urljoin('s3://mybucket', 'temp_file.npz')
    's3://mybucket/temp_file.npz'
    """

    return "/".join(map(lambda x: str(x).rstrip('/') if not str(x).endswith(':/') else str(x), args))


def get_bucket(region,
               bucket_name,
               **kwargs):
    """
    Get a bucket from a region
    :param region:
    :param bucket_name:
    :param kwargs: can be used to specify AWS access keys (if they are not in .boto or .aws, or the environment)
        important key names aws_access_key_id, aws_secret_access_key
    :return:
    >>> get_bucket(_test_region, _test_bucket)
    <Bucket: ls-annotations>
    """

    # aws_access_key_id=access, aws_secret_access_key=secret
    from boto.s3 import connect_to_region
    conn = connect_to_region(region, **kwargs)
    try:
        return conn.get_bucket(bucket_name)
    except Exception as e:
        raise ValueError('{} bucket could not be found, available buckets: {}'.format(bucket_name,
                                                                                      conn.get_all_buckets()))


def list_from_bucket(region, bucket_name, **kwargs):
    # type: (...) -> List[Tuple[str, boto.s3.key.Key]]
    """
    Get a list from each bucket with the path and the actual key (can be downloaded / moved, ...)
    :param region:
    :param bucket_name:
    :param kwargs:
    :return:
    >>> out_list = list_from_bucket(_test_region, _test_bucket)
    >>> print(next(out_list))
    ('s3://ls-annotations/TestAws.zip', <Key: ls-annotations,TestAws.zip>)
    """
    out_bucket = get_bucket(region, bucket_name, **kwargs)
    for c_item in out_bucket.list():
        yield (urljoin('s3://%s' % bucket_name, c_item.name), c_item)


def get_url_as_key(region, in_url, **kwargs):
    """
    A function for easily parsing a url and creating a key from it
    :param region:
    :param in_url:
    :param kwargs:
    :return:
    >>> out_key = get_url_as_key(_test_region, _test_url)
    >>> print(out_key)
    <Key: ls-output,/20170927/TNM_StagingD20170411T141157857101_by_ThomasW_of_ACC23381854/.run_success>
    >>> out_key.content_length
    '1293'
    """
    p_url = urlparse(in_url)
    assert p_url.scheme == 's3', "Only s3 paths are supported"
    c_bucket = get_bucket(region, p_url.netloc, **kwargs)
    return c_bucket.get_key(p_url.path)


def _dummy_show_lines(in_k, n=2):
    with in_k as b:
        for _, line in zip(range(n), b.readlines()):
            print(line.strip())


class with_url_as_tempfile(object):
    """
    Context manager for an s3-based file so it's usable with "with" statement, and cleaned up afterwards
    >>> k = with_url(_test_region, _test_url, mode = 'r')
    >>> _dummy_show_lines(k)
    [NbConvertApp] Converting notebook staging.ipynb to html
    [NbConvertApp] Executing notebook with kernel: python3
    """

    def __init__(self, region,
                 in_url,
                 mode='r',  # type: Optional[str]
                 **kwargs):
        """

        :param region:
        :param in_url:
        :param mode: mode is the mode to open the file with, or none to return the path (which will be deleted)
        :param kwargs:
        """
        c_key = get_url_as_key(region, in_url, **kwargs)
        self.t_file = tempfile.NamedTemporaryFile(delete=False)
        with self.t_file as f:
            c_key.get_file(f)
        if mode is not None:
            self.nt_file = open(self.t_file.name, mode)
        else:
            self.nt_file = self.t_file.name  # just the path

    def __enter__(self):
        return self.nt_file

    def __exit__(self, exc_type, exc_value, traceback):
        if not isinstance(self.nt_file, str):
            self.nt_file.close()
        os.remove(self.t_file.name)


def get_url_as_path(region, in_url, out_path, **kwargs):
    c_key = get_url_as_key(region, in_url, **kwargs)
    with open(out_path, 'wb') as w:
        c_key.get_file(w)
    return out_path