# file: 3d_maxtree.py - supplementary functionality for the SAP package
# code origins: SAP package: Florent Guiotte https://gitlab.inria.fr/fguiotte/sap/-/tree/master
# author David Hansen <david.hansen@stud.plus.ac.at>
# date 08/29/2024 

import higra as hg
import numpy as np 
import sap
from sap.utils import * 
import pprint
from pprint import pformat
import inspect
import tempfile
from pathlib import Path
import idefix
from idefix.utils import * 
import matplotlib.pyplot as plt

    
def get_3D_6_adjacency_graph(shape):
    neighbours = np.array((( 0,  0, -1),
                           ( 0,  0,  1),
                           ( 0, -1,  0),
                           ( 0,  1,  0),
                           (-1,  0,  0),
                           ( 1,  0,  0)),
                          dtype=np.int64)
    graph_implicit = hg.get_nd_regular_implicit_graph(shape, neighbours)
    hg.CptGridGraph.link(graph_implicit, shape)
    hg.set_attribute(graph_implicit, "no_border_vertex_out_degree", 6)
    
    graph = graph_implicit.as_explicit_graph()

    hg.CptGridGraph.link(graph, hg.CptGridGraph.get_shape(graph_implicit))
    hg.set_attribute(graph, "no_border_vertex_out_degree",
                     hg.get_attribute(graph_implicit, "no_border_vertex_out_degree"))

    return graph

def get_3D_26_adjacency_graph(shape):
    neighbors = np.array(((0, 0, -1), (0, 0, 1),(0, -1, 0), (0, 1, 0),(-1, 0, 0), (1, 0, 0),
                          # Plane neighbors
                          (-1, -1,  0), (-1,  1,  0), ( 1, -1,  0), ( 1,  1,  0),( 0, -1, -1),
                          ( 0, -1,  1), ( 0,  1, -1), ( 0,  1,  1),(-1,  0, -1), (-1,  0,  1),
                          ( 1,  0, -1), ( 1,  0,  1),
                          # Diagonal neighbors
                          (-1, -1, -1), (-1, -1,  1), (-1,  1, -1), (-1,  1,  1),
                          ( 1, -1, -1), ( 1, -1,  1), ( 1,  1, -1), ( 1,  1,  1)),
                          dtype=np.int64)
    
    graph_implicit = hg.get_nd_regular_implicit_graph(shape, neighbors)
    hg.CptGridGraph.link(graph_implicit, shape)
    hg.set_attribute(graph_implicit, "no_border_vertex_out_degree", 6)
    
    graph = graph_implicit.as_explicit_graph()

    hg.CptGridGraph.link(graph, hg.CptGridGraph.get_shape(graph_implicit))
    hg.set_attribute(graph, "no_border_vertex_out_degree",
                     hg.get_attribute(graph_implicit, "no_border_vertex_out_degree"))

    return graph


class Tree:
    def __init__(self, image, adjacency, image_name=None, operation_name='non def'):
        if self.__class__ == Tree:
            raise TypeError('Do not instantiate directly abstract class Tree.')

        self._image_name = image_name
        self._image_hash = sap.utils.ndarray_hash(image) if image is not None else None
        self._adjacency = adjacency
        self._image = image
        self.operation_name = operation_name

        if image is not None:
            self._graph = self._get_adjacency_graph()
            self._construct()

    def __str__(self):
        return str(self.__repr__())

    def get_params(self):
        return {'image_name': self._image_name,
                'image_hash': self._image_hash,
                'adjacency': self._adjacency}

    def __repr__(self):
        if hasattr(self, '_tree'):
            rep = self.get_params()
            rep.update({'num_nodes': self.num_nodes(),
                    'image.shape': self._image.shape,
                    'image.dtype': self._image.dtype})
        else:
            rep = {}
        return self.__class__.__name__ + pprint.pformat(rep)

    def _get_adjacency_graph(self):
        if self._adjacency == 4:
            return hg.get_4_adjacency_graph(self._image.shape)
        elif self._adjacency == 8:
            return hg.get_8_adjacency_graph(self._image.shape)
        elif self._adjacency == 6:
            return get_3D_6_adjacency_graph(self._image.shape)
        elif self._adjacency == 26: 
            return get_3D_26_adjacency_graph(self._image.shape)
        else:
            raise NotImplementedError('adjacency of {} is not '
                    'implemented.'.format(self._adjacency))

    def available_attributes(self=None):
        """
        Return a dictionary of available attributes and parameters.

        Returns
        -------
        dict_of_attributes : dict
            The names of available attributes and parameters required.
            The names are keys (str) and the parameters are values (list
            of str) of the dictionary.

        See Also
        --------
        Tree.get_attribute : Return the attribute values of the tree nodes.

        Notes
        -----
        The list of available attributes is generated dynamically. It is
        dependent of higra's installed version. For more details, please
        refer to `higra documentation
        <https://higra.readthedocs.io/en/stable/python/tree_attributes.html>`_
      a  according to the appropriate higra's version.

        Example
        -------
        >>> sap.Tree.available_attributes()
        {'area': ['vertex_area=None', 'leaf_graph=None'],
         'compactness': ['area=None', 'contour_length=None', ...],
         ...
         'volume': ['altitudes', 'area=None']}

        """
        return available_attributes()

    def get_attribute(self, attribute_name, **kwargs):
        """
        Get attribute values of the tree nodes.

        Parameters
        ------
        attribute_name : str
            Name of the attribute (e.g. 'area', 'compactness', ...)

        Returns
        -------
        attribute_values: ndarray
            The values of attribute for each nodes.

        See Also
        --------
        available_attributes : Return the list of available attributes.

        Notes
        -----
        Some attributes require additional parameters. Please refer to
        `available_attributes`. If not stated, some additional
        parameters are automatically deducted. These deducted parameters
        are 'altitudes' and 'vertex_weights'.

        The available attributes depends of higra's installed version.
        For further details Please refer to `higra documentation
        <https://higra.readthedocs.io/en/stable/python/tree_attributes.html>`_
        according to the appropriate higra's version.

        Examples
        --------
        >>> image = np.arange(20 * 50).reshape(20, 50)
        >>> t = sap.MaxTree(image)
        >>> t.get_attribute('area')
        array([   1.,    1.,    1., ...,  998.,  999., 1000.])

        """
        try:
            compute = getattr(hg, 'attribute_' + attribute_name)
        except AttributeError:
            raise ValueError('Wrong attribute or out feature: \'{}\'')

        if 'altitudes' in inspect.signature(compute).parameters:
            kwargs['altitudes'] = kwargs.get('altitudes', self._alt)

        if 'vertex_weights' in inspect.signature(compute).parameters:
            kwargs['vertex_weights'] = kwargs.get('vertex_weights', self._image)

        return compute(self._tree, **kwargs)

    def reconstruct(self, deleted_nodes=None, feature='altitude',
                    filtering='direct'):
        """
        Return the reconstructed image according to deleted nodes.

        Parameters
        ----------
        deleted_nodes : ndarray or boolean, optional
            Boolean array of nodes to delete. The length of the array should be
            of same of node count.
        feature : str, optional
            The feature to be reconstructed. Can be any attribute of the
            tree (see :func:`available_attributes`). The default is
            `'altitude'`, the grey level of the node.
        filtering : str, optional
            The filtering rule to use. It can be 'direct', 'min', 'max' or
            'subtractive'. Default is 'direct'.

        Returns
        -------
        filtered_image : ndarray
            The reconstructed image.

        Examples
        --------
        >>> image = np.arange(5 * 5).reshape(5, 5)
        >>> mt = sap.MaxTree(image)

        >>> mt.reconstruct()
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])

        >>> area = mt.get_attribute('area')

        >>> mt.reconstruct(area > 10)
        array([[ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])


        """
        if isinstance(deleted_nodes, bool):
            deleted_nodes = np.array((deleted_nodes,) * self.num_nodes())
        elif deleted_nodes is None:
            deleted_nodes = np.zeros(self.num_nodes(), dtype=bool)

        feature_value = self._alt if feature == 'altitude' else \
                        self.get_attribute(feature) if isinstance(feature, str) \
                        else feature

        rules = {'direct': self._filtering_direct,
                 'min': self._filtering_min,
                 'max': self._filtering_max,
                 'subtractive': self._filtering_subtractive}

        feature_value, deleted_nodes = rules[filtering](feature_value, deleted_nodes)

        return hg.reconstruct_leaf_data(self._tree, feature_value, deleted_nodes)

    def _filtering_direct(self, feature_value, direct):
        deleted_nodes = direct.astype(bool)
        return feature_value, deleted_nodes

    def _filtering_min(self, feature_value, direct):
        deleted_nodes = hg.propagate_sequential(self._tree, direct,
                ~direct).astype(bool)
        return feature_value, deleted_nodes

    def _filtering_max(self, feature_value, direct):
        deleted_nodes = hg.accumulate_and_min_sequential(self._tree, direct,
                    np.ones(self._tree.num_leaves()),
                    hg.Accumulators.min).astype(bool)
        return feature_value, deleted_nodes

    def _filtering_subtractive(self, feature_value, direct):
        deleted_nodes = direct.astype(bool)
        delta = feature_value - feature_value[self._tree.parents()]
        delta[direct] = 0
        delta[self._tree.root()] = feature_value[self._tree.root()]
        feature_value = hg.propagate_sequential_and_accumulate(self._tree, delta,
                                                             hg.Accumulators.sum)
        return feature_value, deleted_nodes

    def num_nodes(self):
        """
        Return the node count of the tree.

        Returns
        -------
        nodes_count : int
            The node count of the tree.

        """
        return self._tree.num_vertices()

class MaxTree(Tree):
    """
    Max tree class, the local maxima values of the image are in leafs.

    Parameters
    ----------
    image : ndarray
        The image to be represented by the tree structure.
    adjacency : int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.

    Notes
    -----
    Inherits all methods of :class:`Tree` class.

    """
    def __init__(self, image, adjacency=4, image_name=None):
        super().__init__(image, adjacency, image_name, 'thickening')

    def _construct(self):
        self._tree, self._alt = hg.component_tree_max_tree(self._graph, self._image)

