import time
import inspect
from itertools import chain
from .utils import TypeTool
from typing import List, Any, Dict, Optional, Iterable

append_dict = lambda i_dict, **kwargs: dict(list(i_dict.items()) + list(kwargs.items()))
#TODO fix type type : (Dict[str, Any]) -> Dict[str, Any]
append_dict_raw = lambda i_dict, o_dict: dict(list(i_dict.items()) + list(o_dict.items()))
#TODO fix type type : (Dict[str, Any], Dict[str, Any]) -> Dict[str, Any]
cflatten = lambda x: list(chain(*x))
from collections import defaultdict
import warnings

class LocalRDD(object):
    """
    A simple, non-lazy version of the RDD from pyspark for testing purposes

    >>> a = LocalSparkContext().parallelize([1,2,3,4])
    >>> a.first()
    1
    >>> a.map(lambda x: x+1).first()
    2
    >>> a.groupBy(lambda x: x % 2).first()
    (0, [2, 4])
    >>> a.flatMap(lambda x: range(x)).count()
    10
    >>>

    """
    def __init__(self,
                 items, # type: Iterable[Any]
                 prev, # type: LocalRDD
                 command, # type: str
                 code='', # type: str
                 calc_time=None, # type: Optional[float]
                 verbose = False,
                 **args):
        self.items = list(items)
        self.prev = prev
        self.command_args = (command, args)
        self.code = code
        self.verbose = verbose
        if calc_time is not None: self.calc_time = calc_time
        if verbose:
            print("Creating new RDD[{}] from {} with {} entries".format(self.type, command, len(self.items)))

    def first(self):
        return self.items[0]

    def collect(self):
        return self.items

    def count(self):
        return len(self.items)

    def _transform(self, op_name, in_func, lapply_func, **args):
        """
        Args:
            in_func is the function the user supplied
            lapply_func is the function to actually apply
        """
        try:
            trans_func_code = inspect.getsourcelines(in_func)
        except:
            trans_func_code = ''
        stime = time.time()
        new_list = lapply_func(self.items)
        etime = time.time() - stime
        return LocalRDD(new_list, [self], op_name, apply_func=in_func,
                        calc_time=etime,
                        code=trans_func_code,
                        verbose = self.verbose, **args)

    def map(self, apply_func):
        return self._transform('map', apply_func,
                               lapply_func=lambda x_list: [apply_func(x) for x in x_list])

    def mapValues(self, apply_func):
        return self._transform('mapValues', apply_func,
                               lapply_func=lambda x_list: [(k, apply_func(v)) for (k, v) in x_list])

    def values(self):
        return self._transform('values', lambda x: x,
                               lapply_func=lambda x_list: [v for (k, v) in x_list])

    def flatMap(self, apply_func):
        return self._transform('flatMap', apply_func,
                               lapply_func=lambda x_list: cflatten([apply_func(x) for x in x_list]))

    def flatMapValues(self, apply_func):
        return self._transform('flatMapValues', apply_func,
                               lapply_func=lambda x_list: cflatten(
                                   [[(k, y) for y in apply_func(v)] for (k, v) in x_list]))

    def groupBy(self, apply_func):
        """

        :param apply_func:
        :return:

        >>> LocalSparkContext().parallelize([1,2,3,4]).groupBy(lambda x: x % 2).first()
        (0, [2, 4])
        """
        def gb_func(x_list):
            o_dict = defaultdict(list)
            for i in x_list:
                o_dict[apply_func(i)] += [i]
            return list(o_dict.items())

        return self._transform('groupBy', apply_func,
                               lapply_func=gb_func)

    def filter(self, apply_func):
        return self._transform('filter', apply_func,
                               lapply_func=lambda x_list: filter(apply_func, x_list))

    def saveAsPickleFile(self, filename):
        return self._transform('saveAsPickleFile', NamedLambda('SaveFileShard', lambda x: x),
                               lapply_func=lambda x_list: [(x, filename) for x in x_list])

    def repartition(self, *args):
        warnings.warn("Partitioning not really supported yet", RuntimeWarning)
        return self

    def partitionBy(self, *args, **kwargs):
        warnings.warn("Partitioning not really supported yet", RuntimeWarning)
        return self

    def mapPartitions(self, apply_func):
        warnings.warn("Partitioning not really supported yet", RuntimeWarning)
        return self._transform('groupBy', apply_func,
                               lapply_func=lambda x_list: apply_func(x_list))

    def zipWithIndex(self):
        return self._transform('zipWithIndex', zip,
                               lapply_func=lambda x_list: [(x,i) for i, x in enumerate(x_list)])

    def zipWithUniqueId(self):
        return self.zipWithIndex()

    @property
    def type(self):
        return TypeTool.info(self.first())

    @property
    def key_type(self):
        return TypeTool.info(self.items[0][0])

    @property
    def value_type(self):
        return TypeTool.info(self.items[0][1])

    def id(self):
        return (str(self.items), tuple(self.prev), str(self.command_args)).__hash__()


class LocalSparkContext(object):
    """
    A fake spark context for testing/debugging purposes

    >>> a = LocalSparkContext()
    >>> b = a.parallelize([1,2,3,4])
    >>> b.first()
    1
    """
    def __init__(self, verbose = False):
        self.verbose = verbose
        pass

    def parallelize(self, in_list, parts = 1, **kwargs):
        return LocalRDD(in_list, [], 'parallelize', in_list=in_list, verbose = self.verbose)

class LocalSQLContext(object):
    """
    A fake SQL context
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("LocalSQLContext has not been implemented yet and probably will not behave properly",
                      RuntimeWarning)
        ##TODO implement functionality using pandas

class NamedLambda(object):
    """
    allows the use of named anonymous functions for arbitrary calculations
    """

    def __init__(self, code_name, code_block, **add_args):
        """
        Create a namedlambda function
        :param code_name: str the name/description of the code to run
        :param code_block: the function to call when the namedlambda is called
        :param add_args: the additional arguments to give to it
        """
        self.code_name = code_name
        self.code_block = code_block
        self.add_args = add_args
        self.__name__ = code_name

    def __call__(self, *cur_objs, **kwargs):
        return self.code_block(*cur_objs, **append_dict_raw(kwargs,self.add_args))

    def __repr__(self):
        return self.code_name

    def __str__(self):
        return self.__repr__()


class FieldSelector(object):
    """
    allows the use of named anonymous functions for selecting fields (makes dag's more readable)
    """

    def __init__(self, field_name):
        self.field_name = field_name
        self.__name__ = "Select Field: {}".format(self.field_name)

    def __call__(self, cur_obj):
        try:
            return cur_obj[self.field_name]
        except:
            return cur_obj._asdict()[self.field_name]

    def __repr__(self):
        return __name__

    def __str__(self):
        return self.__repr__()

Row = lambda **kwargs: dict(kwargs)

# type tools from SparkSQL
_infer_type = lambda *args, **kwargs: None
_has_nulltype = lambda *args, **kwargs: False

class F(object):
    @staticmethod
    def udf(func, *args, **kwargs):
        return func

class sq_types(object):
    @staticmethod
    def StructType(*args, **kwargs):
        return dict()

    @staticmethod
    def MapType(*args, **kwargs):
        return sq_types.StructType(*args, **kwargs)

    @staticmethod
    def DoubleType(*args, **kwargs):
        return sq_types.StructType(*args, **kwargs)

    @staticmethod
    def FloatType(*args, **kwargs):
        return sq_types.DoubleType(*args, **kwargs)

    @staticmethod
    def IntegerType(*args, **kwargs):
        return sq_types.DoubleType(*args, **kwargs)

    @staticmethod
    def ArrayType(*args, **kwargs):
        return sq_types.StructType(*args, **kwargs)

    @staticmethod
    def StringType(*args, **kwargs):
        return sq_types.StructType(*args, **kwargs)
