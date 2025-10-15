import numpy as np
from collections import defaultdict
import warnings
import time
import sys

class MeshClass:

    # Element type definitions from CGNS standard
    ELEMENT_NAMES_FROM_TYPES = {
        '10': 'TETRA_4',
        '12': 'PYRA_5',
        '14': 'PENTA_6',
        '17': 'HEXA_8',
        '5' : 'TRI_3',
        '7' : 'QUAD_4',
        '2' : 'BAR_2'
    }

    # Element type definitions from CGNS standard
    ELEMENT_TYPES_FROM_NAMES = {
        'TETRA_4': 10,
        'PYRA_5': 12,
        'PENTA_6': 14,
        'HEXA_8': 17,
        'TRI_3': 5,
        'QUAD_4': 7,
        'BAR_2': 2,
        'tri': 5,
        'quad': 7,
        'line': 2
    }

    # Nodes per element for each type
    NODES_PER_ELEMENT = {
        2: 2,    # LINE_2
        5: 3,    # TRI_3
        7: 4,    # QUAD_4
        10: 4,   # TETRA_4
        12: 5,   # PYRA_5
        14: 6,   # PENTA_6
        17: 8,   # HEXA_8
    }

    # Node-to-node Connections per element for each type
    NODESCONNECTIONS_PER_ELEMENT = {
        5: [[1, 2], [0, 2], [0, 1]],    # TRI_3
        7: [[1, 3], [0, 2], [1, 3], [0, 2]],    # QUAD_4
        10: [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],   # TETRA_4
        12: [[1, 3, 4], [0, 2, 4], [1, 3, 4], [0, 2, 4], [0, 1, 2, 3]],   # PYRA_5
        14: [[1, 2, 3], [0, 2, 4], [0, 1, 5], [0, 4, 5], [1, 3, 5], [2, 3, 4]],   # PENTA_6
        17: [[1, 3, 4], [0, 2, 5], [1, 3, 6], [0, 2, 7], [0, 5, 7], [1, 4, 6], [2, 5, 7], [3, 4, 6]],   # HEXA_8  As per CGNS standard
    }

    # Edges shared by the same Node per element for each type
    EDGESOFPOINTS_PER_ELEMENT = {
        5: [[0, 2], [0, 1], [1, 2]],    # TRI_3
        7: [[0, 3], [0, 1], [1, 2], [2, 3]],    # QUAD_4
        10: [[], [], [], []],   # TETRA_4
        12: [[], [], [], [], []],   # PYRA_5
        14: [[], [], [], [], [], []],   # PENTA_6
        17: [[0, 4, 11], [0, 1, 8], [1, 2, 9], [2, 3, 10], 
             [5, 8, 11], [5, 6, 8], [6, 7, 9], [7, 8, 10]],   # HEXA_8  As per CGNS standard
    }

    # Edges as NodeI-NodeJ in local indices per element for each type
    EDGES_PER_ELEMENT = {
        5: [[0, 1], [1, 2], [2, 0]],    # TRI_3
        7: [[0, 1], [1, 2], [2, 3], [3, 0]],    # QUAD_4
        10: [],   # TETRA_4
        12: [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 4], [2, 4], [3, 4]],   # PYRA_5
        14: [],   # PENTA_6
        17: [[0, 1], [1, 2], [2, 3], [3, 0], 
             [4, 5], [5, 6], [6, 7], [7, 4], 
             [1, 5], [2, 6], [3, 7], [0, 4]],   # HEXA_8
    }



    # Faces (as collection of nodes) associated to each element for each type (3D Only)
    FACES_PER_ELEMENT = {
        5: [],    # TRI_3
        7: [],    # QUAD_4
        10: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],   # TETRA_4
        12: [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],   # PYRA_5
        14: [],   # PENTA_6
        17: [[0, 1, 2, 3], [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5],
             [4, 5, 6, 7], [2, 3, 7, 6]],   # HEXA_8
    }


    # Mask for which face is associated to an edge of the element for each type (3D Only)
    FACESOFEDGES_PER_ELEMENT = {
        5: [],    # TRI_3
        7: [],    # QUAD_4
        10: [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],   # TETRA_4
        12: [[], [], [], [],
             [], [], [], []],   # PYRA_5
        14: [],   # PENTA_6
        17: [[0, 1], [0, 3], [0, 5], [0, 2], 
             [4, 1], [4, 3], [4, 5], [4, 4], 
             [1, 3], [3, 5], [2, 5], [1, 2]],   # HEXA_8
    }

    def __init__(self, Logger, GridFileName=None, nDim=3):
        """
        Initialize Element Class

        Parameters:
        -----------
        cgns_file : str, optional
            Path to CGNS file
        """
        self.GridFileName = GridFileName
        self.nDim = nDim
        self.Elements = {}
        self.Nodes = None
        self.NodesConnectedToNode = None
        self.CellsConnectedToNode = None
        self.isNodeOnBoundary = None
        self.boundaryNormals = None
        self.BCsAssociatedToNode = None
        self.n_nodes = 0
        self.boundaries = {}
        self.boundary_faces = {}
        self.boundary_nodes = {}
        self.cell_types = None
        self.ControlFaceDictPerEdge = {}
        self.ControlFaceNormalDictPerEdge = {}
        self.ControlVolumesPerNode = None
        self.boundaryCVArea = None
        self.Logger = Logger
        

    def _compute_line_length(self, nodes, toPrint=True):
        """Compute length for multiple lines"""
        if toPrint:
            self.Logger.info("Compute length for multiple lines...")
        d1 = np.squeeze(nodes[:, 1, :]-nodes[:, 0, :])

        return np.linalg.norm(d1, axis=1)
    
    def _compute_CV_contrib_line(self, nodes):
        """Compute boundary CV area for multiple lines"""
        
        lineLengths = self._compute_line_length(nodes)
        return np.repeat(lineLengths[..., np.newaxis], 2, axis=1)/2
    

    def _compute_single_line_length(self, nodes):
        """Compute length for single line"""
        d1 = np.squeeze(nodes[1, :]-nodes[0, :])

        return np.linalg.norm(d1)
    
    def _compute_tria_area(self, nodes, toPrint=True):
        """Compute area for multiple triangles"""
        if toPrint:
            self.Logger.info("Compute area for multiple triangles...")

        d1 = np.squeeze(nodes[:, 1, :]-nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 2, :]-nodes[:, 1, :])

        return 0.5*np.linalg.norm(np.cross(d1, d2, axis=1), axis=1)

    
    def _compute_CV_contrib_tria(self, nodes):
        """Compute boundary CV area for multiple triangles"""
        self.Logger.info("Compute area for multiple triangles...")

        centroid = np.mean(nodes, axis=1)
        nodesOfPoints = self.NODESCONNECTIONS_PER_ELEMENT.get(5, 0)

        CVAreas = np.zeros((nodes[:, :, 0].shape), dtype=float)
        Points = np.zeros((len(nodes[:, 0, 0]), 4, 3))
        Points[:, 2, :] = centroid

        for iNode in range(3):
            Points[:, 0, :] = nodes[:, iNode, :]
            edges = nodesOfPoints[iNode]
            Points[:, 1, :] = (nodes[:, edges[0], :]+nodes[:, iNode, :])/2
            Points[:, 3, :] = (nodes[:, edges[1], :]+nodes[:, iNode, :])/2
            # Compute CVAreas
            CVAreas[:, iNode] = self._compute_quad_area(Points)

        return CVAreas

    def _compute_single_tria_area(self, nodes):
        """Compute area and for single triangle"""

        d1 = np.squeeze(nodes[1, :]-nodes[0, :])
        d2 = np.squeeze(nodes[2, :]-nodes[1, :])

        return 0.5*np.linalg.norm(np.cross(d1, d2))
    
    def _compute_quad_area(self, nodes):
        """Compute area for quadrilater"""
        self.Logger.info("Compute area for quadrilater...")
        volume = 0.0
        # Split into 2 trias
        tria = [
            [0, 1, 2],
            [2, 3, 0]
        ]
        for tria in tria:
            tria_nodes = nodes[:, tria, :]
            v = self._compute_tria_area(tria_nodes)
            volume += v
        return volume
    

    def _compute_CV_contrib_quad(self, nodes):
        """Compute boundary CV area for multiple quads"""
        self.Logger.info("Compute boundary CV area for multiple quads...")


        centroid = np.mean(nodes, axis=1)
        nodesOfPoints = self.NODESCONNECTIONS_PER_ELEMENT.get(7, 0)
        
        CVAreas = np.zeros((nodes[:, :, 0].shape), dtype=float)
        Points = np.zeros((len(nodes[:, 0, 0]), 4, 3))
        Points[:, 2, :] = centroid

        for iNode in range(4):
            Points[:, 0, :] = nodes[:, iNode, :]
            edges = nodesOfPoints[iNode]
            Points[:, 1, :] = (nodes[:, edges[0], :]+nodes[:, iNode, :])/2
            Points[:, 3, :] = (nodes[:, edges[1], :]+nodes[:, iNode, :])/2
            
            # Compute CVAreas
            CVAreas[:, iNode] = self._compute_quad_area(Points)

        return CVAreas
    
    
    def _compute_tetra_volume(self, nodes, toPrint=True):
        """Compute volume for tetrahedron"""
        if toPrint:
            self.Logger.info("Compute volume for tetrahedrons...")
        v1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        v2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        v3 = np.squeeze(nodes[:, 3, :] - nodes[:, 0, :])
        volume = abs(np.sum(v1*np.cross(v2, v3, axis=1), axis=1)) / 6.0

        return volume
    
    def _compute_single_tetra_volume(self, nodes):
        """Compute volume for tetrahedron"""
        v1 = nodes[1, :] - nodes[0, :]
        v2 = nodes[2, :] - nodes[0, :]
        v3 = nodes[3, :] - nodes[0, :]
        volume = abs(np.sum(v1*np.cross(v2, v3))) / 6.0

        return volume
    

    def _compute_hexa_volume(self, nodes):
        """Compute volume for hexahedron"""
        self.Logger.info("Compute volume for hexahedrons...")
        volume = 0.0
        # Split into 5 tetrahedra
        tets = [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [1, 3, 4, 6],
            [3, 4, 6, 7]
        ]
        for tet in tets:
            tet_nodes = nodes[:, tet, :]
            v = self._compute_tetra_volume(tet_nodes)
            volume += v
        return volume

    def _compute_pyramid_volume(self, nodes):
        """Compute volume for pyramid"""
        self.Logger.info("Compute volume for pyramids...")
        base_center = np.mean(nodes[:, :4, :], axis=1)
        apex = np.squeeze(nodes[:, 4, :])
        d1 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 3, :] - nodes[:, 1, :])
        base_area = 0.5 * np.linalg.norm(np.cross(d1, d2, axis=1), axis=1)
        height = np.linalg.norm(apex - base_center, axis=1)
        volume = base_area * height / 3.0
        return volume

    def _compute_prism_volume(self, nodes):
        """Compute volume for prism (wedge)"""
        self.Logger.info("Compute volume for prisms (wedges)...")
        v1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        v2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        base_area = 0.5 * np.linalg.norm(np.cross(v1, v2, axis=1), axis=1)
        h1 = np.linalg.norm(np.squeeze(nodes[:, 3, :] - nodes[:, 0, :]), axis=1)
        h2 = np.linalg.norm(np.squeeze(nodes[:, 4, :] - nodes[:, 1, :]), axis=1)
        h3 = np.linalg.norm(np.squeeze(nodes[:, 5, :] - nodes[:, 2, :]), axis=1)
        height = (h1 + h2 + h3) / 3.0
        volume = base_area * height
        return volume

    def _compute_normal_quad(self, nodes):
        d1 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 3, :] - nodes[:, 1, :])
        normal = np.cross(d1, d2, axis=1)
        return normal/np.repeat(np.linalg.norm(normal, axis=1)[..., np.newaxis], 3, axis=1)

    def _compute_normal_tria(self, nodes):
        d1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        normal = np.cross(d1, d2, axis=1)
        return normal/np.repeat(np.linalg.norm(normal, axis=1)[..., np.newaxis], 3, axis=1)
    
    def _compute_normal_line(self, nodes):
        d1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        normal = d1.copy()
        normal[:, 0] = d1[:, 1]
        normal[:, 1] = -d1[:, 0]

        return normal/np.repeat(np.linalg.norm(normal, axis=1)[..., np.newaxis], 3, axis=1)
    

    def _read_cgns_h5py(self, zone_name, BCs):
        """Read CGNS using h5py - NumPy 2.0 compatible"""
        try:
            import h5py
        except ImportError:
            self.Logger.error("h5py not installed. Run: pip install h5py")
            raise
        
        with h5py.File(self.GridFileName, 'r') as f:
            # Navigate CGNS tree structure
            base_name = 'Base'
            base = f[base_name]

            if zone_name is None:
                raise ValueError("No zone found in CGNS file")

            zone = base[zone_name]

            # Read grid coordinates
            if 'GridCoordinates' not in zone:
                raise ValueError("GridCoordinates not found in zone")

            grid_coords = zone['GridCoordinates']

            # Read coordinates - handle both old and new format
            def read_coord(name):
                if name in grid_coords:
                    data = grid_coords[name]
                    if ' data' in data:
                        return np.array(data[' data'])
                    else:
                        return np.array(data[()])
                raise ValueError(f"{name} not found")

            x = read_coord('CoordinateX')
            y = read_coord('CoordinateY')
            z = np.zeros(x.shape)
            if self.nDim == 3:
                z = read_coord('CoordinateZ')

            self.Nodes = np.column_stack([x, y, z])
            self.n_nodes = len(self.Nodes)

            isOnBoundary = np.full((len(self.Nodes), 1), False)

            self.Logger.info(f"Loaded {self.n_nodes} nodes")

            AllCell_types = np.zeros((np.array(zone[' data'])[1][0], 1), dtype=int)

            # Read element sections
            for key in zone.keys():
                
                if key.endswith('Elements') or key.endswith('Element') or 'Elements' in key:
                    try:
                        elem_section = zone[key]

                        # Read element type
                        if ' data' in elem_section:
                            elem_type = int(elem_section[' data'][0])
                        else:
                            continue

                        # Read connectivity
                        if 'ElementConnectivity' not in elem_section:
                            continue

                        conn_data = elem_section['ElementConnectivity']
                        if ' data' in conn_data:
                            connectivity = np.array(conn_data[' data'])
                        else:
                            connectivity = np.array(conn_data[()])
                        
                        # Read element range
                        if 'ElementRange' in elem_section:
                            range_data = elem_section['ElementRange']
                            if ' data' in range_data:
                                elem_range = range_data[' data']
                            else:
                                elem_range = range_data[()]
                            start_idx = int(elem_range[0]) - 1
                            end_idx = int(elem_range[1])
                        else:
                            start_idx = 0
                            end_idx = len(connectivity)

                        AllCell_types[start_idx:end_idx] = elem_type
                        nPointsPerElement = self.NODES_PER_ELEMENT[elem_type]
                        connectivity = np.reshape(connectivity-1, (int(len(connectivity)/nPointsPerElement), nPointsPerElement))

                        # Store elements
                        section_name = self.ELEMENT_NAMES_FROM_TYPES[str(elem_type)]
                        self.Elements[section_name] = {
                            'type': elem_type,
                            'connectivity': connectivity,  # Convert to 0-based
                            'range': (start_idx, end_idx)
                        }

                        self.Logger.info(f"  Section '{section_name}': type={elem_type}, "
                              f"elements={end_idx - start_idx}")
                    except Exception as e:
                        self.Logger.error(f"  Warning: Could not read section '{key}': {e}")
                        continue
            
            self.cell_types = AllCell_types

            # Read boundary conditions        
            for bc_name in BCs.keys():
                try:
                    bc = zone[bc_name]
                    # Read boundary face/node list
                    if 'ElementConnectivity' in bc:  # For Pointwise this is actually the element range! PD
                        pl_data = bc['ElementConnectivity']
                        if ' data' in pl_data:
                            connectivity = np.array(pl_data[' data']) - 1
                        else:
                            connectivity = np.array(pl_data[()]) - 1

                        isOnBoundary[connectivity] = True

                        elem_type = self.ELEMENT_TYPES_FROM_NAMES[BCs[bc_name]['Elem_type']]
                        nPointsPerElement = self.NODES_PER_ELEMENT[elem_type]
                        connectivity = np.reshape(connectivity, (int(len(connectivity)/nPointsPerElement), nPointsPerElement))
                        
                        if BCs[bc_name]['Elem_type'] == 'line':
                            normals = self._compute_normal_line(self.Nodes[connectivity])
                            areas = self._compute_line_length(self.Nodes[connectivity])
                            CVAreas = self._compute_CV_contrib_line(self.Nodes[connectivity])
                        elif BCs[bc_name]['Elem_type'] == 'tri':
                            normals = self._compute_normal_tria(self.Nodes[connectivity])
                            areas = self._compute_tria_area(self.Nodes[connectivity])
                            CVAreas = self._compute_CV_contrib_tria(self.Nodes[connectivity])
                        elif BCs[bc_name]['Elem_type'] == 'quad':
                            normals = self._compute_normal_quad(self.Nodes[connectivity])
                            areas = self._compute_quad_area(self.Nodes[connectivity])
                            CVAreas = self._compute_CV_contrib_quad(self.Nodes[connectivity])
                        
                        self.boundary_nodes[bc_name] = {
                                                        'connectivity' : connectivity,
                                                        'normals': normals,
                                                        'areas': areas,
                                                        'elem_type': BCs[bc_name]['Elem_type'],
                                                        'BoundaryCVArea': CVAreas
                                                        }
                        self.boundary_faces[bc_name] = bc_name
                        self.Logger.info(f"  Boundary '{bc_name}': {len(connectivity)} nodes")

                except Exception as e:
                    self.Logger.error(f"  Warning: Could not read BC '{bc_name}': {e}")
                    continue
            
            self.isNodeOnBoundary = isOnBoundary


        

    def construct_secondaryMeshStructures(self, boundary_conditions):

        self.Logger.info(f"Constructing NodesConnectedToNode structure...")
        self.SetNodesConnectedToNode()

        self.Logger.info(f"Constructing boundary normals structure...")
        self.set_boundary_Variables(boundary_conditions)

        self.Logger.info(f"Constructing Cell Volumes and Cell Centers structure...")
        self.compute_geometric_properties()

    def SetNodesConnectedToNode(self):
        # Build node-to-node connectivity from elements
        node_neighbors = defaultdict(set)
        node_cells = [[] for _ in range(self.n_nodes)]

        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            nodes_per_elem = self.NODES_PER_ELEMENT.get(elem_type, 0)
            nodesConns_per_elem = self.NODESCONNECTIONS_PER_ELEMENT.get(elem_type, 0)
            if nodes_per_elem == 0:
                continue

            n_elems = len(connectivity[:, 0])

            # Build connectivity
            for iCell in range(n_elems):
                if iCell%10000 == 0:
                    self.Logger.info(f"Cells analyzed: {iCell}")
                # Each node is connected to other nodes in the element
                # following the NODESCONNECTIONS_PER_ELEMENT 
                for iNode, iNodeGlobal in enumerate(connectivity[iCell]):
                    nodeConn = nodesConns_per_elem[iNode]
                    node_neighbors[iNodeGlobal].update(connectivity[iCell, nodeConn])
                    node_cells[iNodeGlobal] = node_cells[iNodeGlobal] + [(iCell, section_name, iNode)]
            

        self.NodesConnectedToNode = node_neighbors
        self.NodesConnectedToNodeTotal = node_neighbors
        self.CellsConnectedToNode = node_cells

    def set_boundary_Variables(self, boundary_conditions):
        # Now assign normals to boundary nodes
            boundaryNormal = np.zeros((len(self.Nodes), 3), dtype=float)
            boundaryCVArea = np.zeros((len(self.Nodes), ), dtype=float)
            BCsAssociatedToNode = [[] for _ in range(self.n_nodes)]
            for iNode in range(self.n_nodes):
                bnormal = np.zeros((1,3), dtype=float)
                bcs = []
                if self.isNodeOnBoundary[iNode]: # Search only among points on boundaries
                    for bname, bdict in self.boundary_nodes.items():
                        bnodes = bdict['connectivity']
                        if iNode in bnodes:
                            bcs = bcs+[bname]
                            bc_name = bname
                            
                            if bname in boundary_conditions:
                                bc_data = boundary_conditions[bname]                                
                                # Then search among all of the elements
                                mask = (bnodes == iNode)
                                boolForElementsWithPoint = (mask).any(axis=1)
                                normals2Include = bdict['normals'][boolForElementsWithPoint]
                                bnormal = np.vstack((bnormal, normals2Include))
                                # print(bdict['BoundaryCVArea'][mask])
                                boundaryCVArea[iNode] += np.sum(bdict['BoundaryCVArea'][mask])
                                
                                # else:
                                #     if not bc_data['value'] == 0:
                                #         print("ERROR! Boundary condition of type", bc_data['value'], "not recognized!")
                                #         exit(1)

                bnormalmean = bnormal
                if len(bnormal[:, 0]) > 1:
                    bnormalmean = np.mean(bnormal[1:], axis=0)
                
                boundaryNormal[iNode, :] = bnormalmean
                BCsAssociatedToNode[iNode] = bcs

            self.boundaryNormal = boundaryNormal
            self.BCsAssociatedToNode = BCsAssociatedToNode
            self.boundaryCVArea = boundaryCVArea

    def compute_geometric_properties(self):
        """
        Compute cell volumes, centers for finite volume method
        """
        self.Logger.info("Computing geometric properties...")

        cell_volumes = np.array([], dtype=float)
        cell_centers = np.array([], dtype=float)

        cell_idx = 0
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
         

            nodes_per_elem = self.NODES_PER_ELEMENT.get(elem_type, 0)
            if nodes_per_elem == 0:
                self.Logger.info(f"  Warning: Unknown element type {elem_type} in '{section_name}'")
                continue

            elem_nodes = self.Nodes[connectivity]

            # Pretty sure I just need the centroid
            centers = np.mean(elem_nodes, axis=1)

            # # Compute volume and center based on element type
            # if   elem_type == 5:  # TRI_3
            #     volumes = self._compute_tria_area(elem_nodes)
            # elif elem_type == 7:  # QUAD_4
            #     volumes = self._compute_quad_area(elem_nodes)
            # elif elem_type == 10:  # TETRA_4
            #     volumes = self._compute_tetra_volume(elem_nodes)
            # elif elem_type == 17:  # HEXA_8
            #     volumes = self._compute_hexa_volume(elem_nodes)
            # elif elem_type == 12:  # PYRA_5
            #     volumes = self._compute_pyramid_volume(elem_nodes)
            # elif elem_type == 14:  # PENTA_6
            #     volumes = self._compute_prism_volume(elem_nodes)
            # else:
            #     # Default approximation for higher-order elements
            #     self.Logger.error("ERROR! Element type", elem_type, "not recognized!")
            #     exit(1)

            # self.Elements[section_name]["CellVolumes"] = volumes
            self.Elements[section_name]["CellCenters"] = centers

    def build_DualControlVolumes(self):
        
        self.Logger.info("-----------------------------")
        if self.nDim == 2:
            self.Logger.info("Building 2D control volumes...")
            self.build_DualControlVolumes_2D()
        if self.nDim == 3:
            self.Logger.info("Building 3D control volumes...")
            self.build_DualControlVolumes_3D()

        self.Logger.info("-----------------------------")


    def build_DualControlVolumes_2D(self):

        # Let's start from computing the control volumes associated 
        # to each node and their elements.

        start = time.time()
        self.Logger.info("Initializing ControlVolumes per Node dictionary...")


        # Construct a dictionary of edges for each point that will have the area associated to it
        ControlVolumesPerNode = np.zeros((self.n_nodes, ), dtype=float)
        ControlFaceDictPerEdge = {}
        ControlFaceNormalDictPerEdge = {}
        for iNode in range(self.n_nodes):

            neighbors = list(self.NodesConnectedToNode[iNode])
            for neigh in neighbors:
                ControlFaceDictPerEdge[str(iNode)+"-"+str(neigh)] = 0.0
                ControlFaceNormalDictPerEdge[str(iNode)+"-"+str(neigh)] = np.array([0.0, 0.0, 0.0])

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")


        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")
        self.Logger.info("REMARK: the face area is the same but the control volume not necessarily!")


        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            edges_per_elem = self.EDGES_PER_ELEMENT.get(elem_type, 0)

            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            edgeCentroids = np.mean(ElementNodes[:, edges_per_elem, :], axis=2)

            self.Elements[section_name]["edgeCentroids"] = edgeCentroids

            faces_per_elem = self.FACES_PER_ELEMENT.get(elem_type, 0)
            faceCentroids = np.mean(ElementNodes[:, faces_per_elem, :], axis=2)
            self.Elements[section_name]["FaceCentroids"] = faceCentroids

            facesOfEdges = self.FACESOFEDGES_PER_ELEMENT.get(elem_type, 0)
            
            SectionalAreaOfEdges = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem)), dtype=float)
            LineNormalsOfEdgeFaces = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem), 3), dtype=float)

            lines = np.zeros((len(elem_data["connectivity"][:, 0]), 2, 3), dtype=float)

            lines[:, 1, :] = elem_data["CellCenters"]

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            for iEdge, edge in enumerate(edges_per_elem):

                lines[:, 0, :] = edgeCentroids[:, iEdge]

                LineNormalsOfEdgeFaces[:, iEdge, :] = self._compute_normal_line(lines)
                SectionalAreaOfEdges[:, iEdge] = self._compute_line_length(lines, toPrint=False)
                
                for iElem in range(len(connectivity[:, 0])):
                    edgeName = "-".join(map(str, connectivity[iElem, edge]))
                    ControlFaceDictPerEdge[edgeName] += SectionalAreaOfEdges[iElem, iEdge]
                    ControlFaceNormalDictPerEdge[edgeName] += LineNormalsOfEdgeFaces[iElem, iEdge, :]

            
            self.Elements[section_name]["SectionalAreaOfEdges"] = SectionalAreaOfEdges
            self.Elements[section_name]["ControlFaceNormalDictPerEdge"] = ControlFaceNormalDictPerEdge


            edgesOfPoints = self.EDGESOFPOINTS_PER_ELEMENT.get(elem_type, 0)
            trias = np.zeros((len(elem_data["connectivity"][:, 0]), 3, 3), dtype=float)
            trias[:, 0, :] = elem_data["CellCenters"]

            for iPoint in range(self.NODES_PER_ELEMENT.get(elem_type, 0)):

                pointsGlobalIndices = elem_data["connectivity"][:, iPoint]

                trias[:, 1, :] = self.Nodes[pointsGlobalIndices, :]

                # Now I compute the areas of each edge midpoint
                # Cycle on each edge
                for iEdge, edge in enumerate(edgesOfPoints[iPoint]):

                    trias[:, 2, :] = edgeCentroids[:, edge]

                    controlVolumeContrib = self._compute_tria_area(trias, toPrint=False)
                    # I can already add it to the global index
                    np.add.at(ControlVolumesPerNode, pointsGlobalIndices, controlVolumeContrib)
        
        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Collecting the edge face area from every element")

        AlreadyProcessed = {}
        for key in ControlFaceDictPerEdge.keys():
            reverseKey = "-".join(list(reversed(key.split("-"))))
            
            # It should not matter if I do it twice , but I exclude already processed edges just to be sure
            if not key in AlreadyProcessed.keys():

                # First extract the two normals
                normal_key = ControlFaceNormalDictPerEdge[key]
                normal_reversekey = ControlFaceNormalDictPerEdge[reverseKey]

                # Then reverse the one from reverse key
                normal_reversekey = -normal_reversekey

                # Then perform weighted average
                face_key = ControlFaceDictPerEdge[key]
                face_reversekey = ControlFaceDictPerEdge[reverseKey]
                newNormal = (normal_key*face_key + normal_reversekey*face_reversekey) / (face_key+face_reversekey)

                # Now update the normal
                ControlFaceNormalDictPerEdge[key] = newNormal
                ControlFaceNormalDictPerEdge[reverseKey] = -newNormal

                # Now compute the areas
                dummy = (face_key + face_reversekey)/2
                ControlFaceDictPerEdge[key] = dummy
                ControlFaceDictPerEdge[reverseKey] = dummy
                
                AlreadyProcessed[key] = True
                AlreadyProcessed[reverseKey] = True

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        
        self.ControlFaceDictPerEdge = ControlFaceDictPerEdge
        self.ControlFaceNormalDictPerEdge = ControlFaceNormalDictPerEdge
        self.ControlVolumesPerNode = ControlVolumesPerNode


    def build_DualControlVolumes_3D(self):

        # Let's start from computing the control volumes associated 
        # to each node and their elements.

        start = time.time()
        self.Logger.info("Initializing ControlVolumes per Node dictionary...")


        # Construct a dictionary of edges for each point that will have the area associated to it
        ControlVolumesPerNode = np.zeros((self.n_nodes, ), dtype=float)
        ControlFaceDictPerEdge = {}
        for iNode in range(self.n_nodes):

            neighbors = list(self.NodesConnectedToNode[iNode])
            for neigh in neighbors:
                ControlFaceDictPerEdge[str(iNode)+"-"+str(neigh)] = 0.0

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")


        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")
        self.Logger.info("REMARK: the face area is the same but the control volume not necessarily!")

        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            edges_per_elem = self.EDGES_PER_ELEMENT.get(elem_type, 0)

            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            edgeCentroids = np.mean(ElementNodes[:, edges_per_elem, :], axis=2)

            self.Elements[section_name]["edgeCentroids"] = edgeCentroids

            faces_per_elem = self.FACES_PER_ELEMENT.get(elem_type, 0)
            faceCentroids = np.mean(ElementNodes[:, faces_per_elem, :], axis=2)
            self.Elements[section_name]["FaceCentroids"] = faceCentroids

            facesOfEdges = self.FACESOFEDGES_PER_ELEMENT.get(elem_type, 0)
            
            SectionalAreaOfEdges = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem)), dtype=float)

            trias = np.zeros((len(elem_data["connectivity"][:, 0]), 3, 3), dtype=float)

            trias[:, 0, :] = elem_data["CellCenters"]

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            for iEdge, edge in enumerate(edges_per_elem):

                trias[:, 1, :] = edgeCentroids[:, iEdge]

                for face in facesOfEdges[iEdge]:
                                            
                    # Compose tria and compute area contribution
                    trias[:, 2, :] = faceCentroids[:, face, :]
                    faceAreaContrib = self._compute_tria_area(trias, toPrint=False)
                    SectionalAreaOfEdges[:, iEdge] += faceAreaContrib
                
                for iElem in range(len(connectivity[:, 0])):
                    edgeName = "-".join(map(str, connectivity[iElem, edge]))
                    ControlFaceDictPerEdge[edgeName] += SectionalAreaOfEdges[iElem, iEdge]

            
            self.Elements[section_name]["SectionalAreaOfEdges"] = SectionalAreaOfEdges


            edgesOfPoints = self.EDGESOFPOINTS_PER_ELEMENT.get(elem_type, 0)
            tetras = np.zeros((len(elem_data["connectivity"][:, 0]), 4, 3), dtype=float)
            tetras[:, 0, :] = elem_data["CellCenters"]

            for iPoint in range(self.NODES_PER_ELEMENT.get(elem_type, 0)):

                pointsGlobalIndices = elem_data["connectivity"][:, iPoint]

                tetras[:, 1, :] = self.Nodes[pointsGlobalIndices, :]

                # Now I compute the areas of each edge midpoint
                # Cycle on each edge
                for iEdge, edge in enumerate(edgesOfPoints[iPoint]):

                    tetras[:, 2, :] = edgeCentroids[:, edge]

                    for face in facesOfEdges[edge]:
                                                
                        # Compose tetra and compute the volume
                        tetras[:, 3, :] = faceCentroids[:, face, :]
                        controlVolumeContrib = self._compute_tetra_volume(tetras, toPrint=False)

                        # I can already add it to the global index
                        np.add.at(ControlVolumesPerNode, pointsGlobalIndices, controlVolumeContrib)

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Collecting the edge face area from every element")

        AlreadyProcessed = {}
        for key in ControlFaceDictPerEdge.keys():
            reverseKey = "-".join(list(reversed(key.split("-"))))
            
            # It should not matter if I do it twice , but I exclude already processed edges just to be sure
            if not key in AlreadyProcessed.keys():
                dummy = (ControlFaceDictPerEdge[key] + ControlFaceDictPerEdge[reverseKey])
                ControlFaceDictPerEdge[key] = dummy
                ControlFaceDictPerEdge[reverseKey] = dummy
                AlreadyProcessed[key] = True
                AlreadyProcessed[reverseKey] = True

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        
        self.ControlFaceDictPerEdge = ControlFaceDictPerEdge
        self.ControlVolumesPerNode = ControlVolumesPerNode

        








        

                
                
                
