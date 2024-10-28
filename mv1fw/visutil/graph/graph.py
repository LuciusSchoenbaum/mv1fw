



from graphviz import Digraph


try:
    from torch import is_tensor as torch_is_tensor
    from torch.autograd import Variable as torch_Variable
except ImportError:
    def torch_is_tensor():
        return False
    class torch_Variable:
        pass



# helper
def strip_attr(attr):
    # Saved attributes for grad_fn,
    # including saved variables, begin with _saved_.
    saved_ = '_saved_'
    if not attr.startswith(saved_):
        return None
    else:
        return attr[len(saved_):]



class Graph:
    """
    Utility to generate visualizations
    of computational graphs for tensors embedded in the
    PyTorch autodifferentiation mechanism (autograd).
    This code (and description) is forked from the small package `torchviz
    <https://github.com/szagoruyko/pytorchviz/tree/master>`_.

    If a node represents a backward function, it is gray. Otherwise, the node
    represents a tensor and is either blue, orange, or green:

     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    .. note::
        This functionality may also be available in pytorch core, see
        `here <https://pytorch.org/docs/stable/tensorboard.html>`_.

    Arguments:

        params (dict of (name, tensor)):
            information needed to add names to node that requires grad
        show_attrs (boolean):
            whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved (boolean):
            whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars (integer): if show_attrs is `True`, sets max number of characters
            to display for any given attribute.

    """

    def __init__(
            self,
            params=None,
            show_attrs=False,
            show_saved=False,
            max_attr_chars=50,
    ):
        self.show_attrs = show_attrs
        self.show_saved = show_saved
        self.max_attr_chars = max_attr_chars
        if params is not None:
            assert all(isinstance(p, torch_Variable) for p in params.values())
            self.param_map = {id(v): k for k, v in params.items()}
        else:
            self.param_map = {}
        self.graph = None
        # temporary for recursive trip through the graph during generation
        self.seen = None


    def init(
            self,
            variable = None,
            variables = None,
    ):
        """
        Normal use accepts a variable,
        but a list or tuple of variables can be passed in.

        Arguments:

            variable: target variable
            variables: tuple or list of variables
        """
        self.graph = Digraph(
            node_attr=dict(
                style='filled',
                shape='box',
                align='left',
                fontsize='10',
                ranksep='0.1',
                height='0.2',
                fontname='Times-Roman',
            ),
            graph_attr=dict(
                size="12,12",
                # this doesn't extend the drawing area.
                # margin="1.0",
                # this works much better.
                pad="1.0",
            ),
        )
        self.seen = set()
        if variable is not None:
            self._add_base_tensor(variable)
        else:
            if variables is not None:
                for variable in variables:
                    self._add_base_tensor(variable)
            else:
                raise ValueError
        self.resize_graph()
        # > clean up
        self.seen = None


    def resize_graph(self, size_per_element=0.15, min_size=12.):
        """
        Resize the graph according to how much content it contains.
        """
        # the approximate number of nodes and edges
        num_rows = len(self.graph.body)
        content_size = num_rows * size_per_element
        size = max(min_size, content_size)
        str_size = str(size) + "," + str(size)
        self.graph.graph_attr.update(size=str_size)


    def store(self, filename):
        """
        Save the graph to disk, as a .png file ITCINOOD.

        Arguments:

            filename: relative or absolute path.
        """
        # PyPinnch makes its own filenames so we have to
        # bobble the handoff slightly
        # -1 g -2 n -3 p -4 .
        filename_ = filename[:-4]
        self.graph.render(filename_, format='png', cleanup=True)


    def dot_source(self):
        """
        Get the dot language representation of the graph.

        """
        return self.graph.source()


    ########################################################
    # private methods


    def _add_base_tensor(
            self,
            variable,
            color='darkolivegreen1'
    ):
        """
        Recursive method


        Arguments:

            variable:
            color:
        """
        if variable in self.seen:
            return
        self.seen.add(variable)
        self.graph.node(str(id(variable)), self._get_variable_name(variable), fillcolor=color)
        if variable.grad_fn:
            self._add_nodes(grad_fn=variable.grad_fn)
            self.graph.edge(str(id(variable.grad_fn)), str(id(variable)))
        if variable._is_view():
            self._add_base_tensor(variable._base, color='darkolivegreen3')
            self.graph.edge(str(id(variable._base)), str(id(variable)), style="dotted")


    def _get_function_name(self, fn, show_attrs, max_attr_chars):
        print_passing_node_info = False
        if print_passing_node_info:
            print(f"[Graph] _get_function_name:")
            print(f"[Graph] name: {fn.name()}")
            print(f"[Graph] type: {type(fn)}")
            print(f"[Graph] dir: {dir(fn)}")
            print(f"[Graph] string: {str(fn)}")
            print(f"[Graph] docstring: {fn.__doc__}")
            print("-----")
        name = str(type(fn).__name__)
        out = name
        if show_attrs:
            attrs = dict()
            for attr in dir(fn):
                sattr = strip_attr(attr)
                if sattr is None:
                    continue
                val = getattr(fn, attr)
                if torch_is_tensor(val):
                    attrs[sattr] = "[saved tensor]"
                elif isinstance(val, tuple) and any(torch_is_tensor(t) for t in val):
                    attrs[sattr] = "[saved tensors]"
                else:
                    attrs[sattr] = str(val)
            if attrs:
                max_attr_chars = max(max_attr_chars, 3)
                col1width = max(len(k) for k in attrs.keys())
                col2width = min(max(len(str(v)) for v in attrs.values()), max_attr_chars)
                sep = "-" * max(col1width + col2width + 2, len(name))
                attrstr = '%-' + str(col1width) + 's: %' + str(col2width)+ 's'
                truncate = lambda s: s[:col2width - 3] + "..." if len(s) > col2width else s
                params = '\n'.join(attrstr % (k, truncate(str(v))) for (k, v) in attrs.items())
                out += '\n' + sep + '\n' + params
        return out


    def _get_variable_name(self, variable, name=None):
        if not name:
            name = self.param_map[id(variable)] if id(variable) in self.param_map else ''
        str_size = '(' + ', '.join(['%d' % v for v in variable.size()]) + ')'
        return '%s\n %s' % (name, str_size)


    def _add_nodes(self, grad_fn):
        x = grad_fn
        assert not torch_is_tensor(x)
        if not x in self.seen:
            self.seen.add(x)
            if self.show_saved:
                for attr in dir(x):
                    sattr = strip_attr(attr)
                    if sattr is None:
                        continue
                    val = getattr(x, attr)
                    self.seen.add(val)
                    if torch_is_tensor(val):
                        self.graph.edge(str(id(x)), str(id(val)), dir="none")
                        self.graph.node(str(id(val)), self._get_variable_name(val, sattr), fillcolor='orange')
                    if isinstance(val, tuple):
                        for i, t in enumerate(val):
                            if torch_is_tensor(t):
                                name = sattr + '[%s]' % str(i)
                                self.graph.edge(str(id(x)), str(id(t)), dir="none")
                                self.graph.node(str(id(t)), self._get_variable_name(t, name), fillcolor='orange')
            if hasattr(x, 'variable'):
                # if grad_accumulator, add the node for `.variable`
                variable = x.variable
                self.seen.add(variable)
                self.graph.node(str(id(variable)), self._get_variable_name(variable), fillcolor='lightblue')
                self.graph.edge(str(id(variable)), str(id(x)))
            # add the node for this grad_fn
            self.graph.node(str(id(x)), self._get_function_name(x, self.show_attrs, self.max_attr_chars))
            # recurse
            if hasattr(x, 'next_functions'):
                for u in x.next_functions:
                    if u[0] is not None:
                        self.graph.edge(str(id(u[0])), str(id(x)))
                        self._add_nodes(u[0])
            # note: this used to show .saved_tensors in pytorch0.2, but stopped
            # working* as it was moved to ATen and Variable-Tensor merged
            # also note that this still works for custom autograd functions
            if hasattr(x, 'saved_tensors'):
                for t in x.saved_tensors:
                    self.seen.add(t)
                    self.graph.edge(str(id(t)), str(id(x)), dir="none")
                    self.graph.node(str(id(t)), self._get_variable_name(t), fillcolor='orange')










