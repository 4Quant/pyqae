import pydot
import time
import numpy as np
from ..utils import TypeTool
from collections import namedtuple as _invisiblenamedtuple

rdd_node_colors = {'parallelize': 'turquoise', 'map': 'lightgreen', 'mapValues': 'lightgreen',
                   'flatMap': 'yellow', 'flatMapValues': 'yellow',
                   'groupBy': 'orange', 'reduceByKey': 'lightred'}


from collections import OrderedDict



def show_dag(cur_rdd, o_file, calc_size=0):
    """
    Shows the DAG for a given RDD in iPython
    :param cur_rdd: the RDD to show the tree for (recurses by parent)
    :param o_file: file to save the SVG to
    :param calc_size: should the sizes be calculated (0 no, 1 just the first, 2 all sizes)
    :return:
    """
    from IPython.display import SVG
    b_dot = trace_dag(cur_rdd, calc_size=calc_size)
    b_dot.write_svg(o_file)
    return SVG(o_file)


def trace_dag(cur_rdd, calc_size=0):
    dot_graph = pydot.Dot(graph_type='digraph')
    dot_graph.set('rankdir', 'TB')
    dot_graph.set('concentrate', False)
    dot_graph.set_colorscheme('paired9')
    dot_graph.set_node_defaults(shape='record')
    add_rdd_node(dot_graph, cur_rdd, calc_size=calc_size)
    return dot_graph


def add_rdd_node(dot_graph, cur_rdd, calc_size=0):
    """
    Recursive function to add all RDDs to a DAG given a starting node
    """

    def make_node(name, label):
        cur_node = pydot.Node(name)
        cur_node.set_label(label)
        dot_graph.add_node(cur_node)
        return cur_node

    def make_link(a_node, b_node, label=None, width=1, style='dashed'):
        cur_edge = pydot.Edge(a_node, b_node)
        cur_edge.set_penwidth(width)
        cur_edge.set_style(style)
        if label is not None: cur_edge.set_label(label)
        dot_graph.add_edge(cur_edge)
        return cur_edge

    try:
        op_name = cur_rdd.command_args[0]
        cmd_name = cur_rdd.command_args[1]
        func_obj = cur_rdd.command_args[1].get('apply_func', lambda x: x)

    except:
        print("must be a real rdd")
        op_name = "MapPartitions"
        func_obj = cur_rdd.__dict__.get('func', lambda x: x)

    func_name = 'Custom' if func_obj.__name__.find('lambda') >= 0 else func_obj.__name__
    func_name = 'Custom' if func_name.find('pipeline_func') >= 0 else func_name
    if op_name == 'parallelize': func_name = 'parallelize : {}'.format(func_name)
    f_time, f_ele = estimate_time(lambda: cur_rdd.first())
    f_count = cur_rdd.count()
    out_label_dict = OrderedDict({'operation': func_name})
    out_label_dict['output type'] = TypeTool.info(f_ele)
    out_label_dict['elements'] = str(f_count)
    out_label_dict['time per element'] = _pprint_time(cur_rdd.__dict__.get('calc_time', f_time * f_count) / f_count)
    if calc_size > 0:
        if calc_size == 1:
            e_size_txt = 'Mean: %s' % estimate_size(f_ele, 1)
            t_size_txt = estimate_size(f_ele, f_count)
        else:
            e_size_txt, t_size_txt = estimate_list_size(cur_rdd.collect())
        out_label_dict['element size'] = e_size_txt
        out_label_dict['total size'] = t_size_txt

    mkeys = out_label_dict.keys()
    mvalues = [out_label_dict[ckey] for ckey in mkeys]

    label = '{{{keys}}}|{{{values}}}'.format(keys="|".join(map(lambda x: "{}:".format(x), mkeys)),
                                             values="|".join(map(str, mvalues)))

    c_node = make_node(cur_rdd.id(), label)
    if rdd_node_colors.get(op_name, None) is not None:
        c_node.set_style('filled')
        c_node.set_fillcolor(rdd_node_colors[op_name])

    if type(cur_rdd) is list:
        parent_nodes = []
    else:
        parent_nodes = cur_rdd.__dict__.get('prev', [])
        if type(parent_nodes) is not list: parent_nodes = [parent_nodes]

    for inode in parent_nodes:
        next_node, next_op_name = add_rdd_node(dot_graph, inode, calc_size=calc_size)
        l_count = 3 if op_name.find('flat') >= 0 else 1
        for l_id in range(l_count):
            make_link(next_node, c_node, label=op_name if l_id is 0 else None)
    return c_node, op_name


def _calc_size(obj, copies=1):
    """
    Calculate the size of an object by pickling it (reasonable approximation for most data)
    Note: the types must be accessible within the global namespace which means particularly for notebooks
    to make sure they are available there
    """
    import pickle
    tsize = len(pickle.dumps(obj)) * copies
    return tsize


def _pprint_size(tsize):
    for i, lab in [(9, 'GB'), (6, 'MB'), (3, 'kB')]:
        if tsize > 10 ** i:
            return '%2.1f%s' % (tsize / 10 ** i, lab)
    return '%d bytes' % tsize


def _pprint_time(ttime):
    for i, lab in [(0, 's'), (-3, 'ms'), (-6, 'ns')]:
        if ttime > 10 ** i:
            return '%2.1f%s' % (ttime / 10 ** i, lab)


def estimate_list_size(lobj):
    all_sizes = np.array([_calc_size(obj, 1) for obj in lobj])
    flist = [('Total', np.sum)]
    elist = [('Min', np.min), ('Mean', np.mean), ('Max', np.max)] if all_sizes.std() > 0 else [('Mean', np.mean)]
    _exp_str = lambda xlist: '{}'.format(
        ", ".join(['{}: {}'.format(cname, _pprint_size(cfunc(all_sizes))) for cname, cfunc in xlist]))
    return _exp_str(elist), _pprint_size(np.sum(all_sizes))


def estimate_size(obj, copies=1):
    return _pprint_size(_calc_size(obj, copies))


def estimate_time(func_call):
    stime = time.time()
    res = func_call()
    return time.time() - stime, res


def namedtuple(*args, **kwargs):
    """
    namedtuple with a global presence for using the pickler
    :param args:
    :param kwargs:
    :return:
    """
    nt_class = _invisiblenamedtuple(*args, **kwargs)
    # put it on the global scale so it can be tupled correctly
    globals()[nt_class.__name__] = nt_class
    return nt_class