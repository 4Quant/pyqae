"""
The code managing the backend (simplespark or pyspark) and making them as interchangable as possible
"""
import warnings

try:
    # These are only available inside of the pyspark application (using py4j)
    from pyspark.sql import Row
    from pyspark.rdd import RDD
    from pyspark import SparkContext
    from pyspark.sql import SQLContext

    from pyspark.sql.types import _infer_type, _has_nulltype
    from pyspark.sql import Row, F
    import pyspark.sql.types as sq_types

except ImportError:
    warnings.warn("Pyspark is not available using simplespark backend instead", ImportWarning)
    from pyqae.simplespark import Row
    from pyqae.simplespark import LocalRDD as RDD
    from pyqae.simplespark import LocalSparkContext as SparkContext
    from pyqae.simplespark import LocalSQLContext as SQLContext
    from pyqae.simplespark import _infer_type, _has_nulltype, sq_types, F

StructType = sq_types.StructType
MapType = sq_types.MapType
ArrayType = sq_types.ArrayType
