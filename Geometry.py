import numpy as np
from collections import defaultdict
import warnings

class MeshClass:

    # Element type definitions from CGNS standard
    ELEMENT_TYPES = {
        'TETRA_4': 10,
        'PYRA_5': 12,
        'PENTA_6': 14,
        'HEXA_8': 17,
        'TETRA_10': 11,
        'PYRA_14': 13,
        'PENTA_15': 15,
        'HEXA_20': 18,
        'HEXA_27': 19,
        'TRI_3': 5,
        'QUAD_4': 7,
        'MIXED': 20,
        'NGON_n': 22,
        'NFACE_n': 23
    }

    # Nodes per element for each type
    NODES_PER_ELEMENT = {
        5: 3,    # TRI_3
        7: 4,    # QUAD_4
        10: 4,   # TETRA_4
        12: 5,   # PYRA_5
        14: 6,   # PENTA_6
        17: 8,   # HEXA_8
        11: 10,  # TETRA_10
        13: 14,  # PYRA_14
        15: 15,  # PENTA_15
        18: 20,  # HEXA_20
        19: 27   # HEXA_27
    }

    # Node-to-node Connections per element for each type
    NODESCONNECTIONS_PER_ELEMENT = {
        5: 3,    # TRI_3
        7: 4,    # QUAD_4
        10: [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],   # TETRA_4
        12: [[1, 3, 4], [0, 2, 4], [1, 3, 4], [0, 2, 4], [0, 1, 2, 3]],   # PYRA_5
        14: [[1, 2, 3], [0, 2, 4], [0, 1, 5], [0, 4, 5], [1, 3, 5], [2, 3, 4]],   # PENTA_6
        # 17: [[1, 3, 4], [0, 2, 3], [1, 3, 6], [0, 2, 7], [0, 5, 7], [1, 4, 6], [1, 2, 5], [3, 4, 6]],   # HEXA_8
        17: [[1, 3, 4], [0, 2, 5], [1, 3, 6], [0, 2, 7], [0, 5, 7], [1, 4, 6], [2, 5, 7], [3, 4, 6]],   # HEXA_8
    }

    def __init__(self, GridFileName=None):
        """
        Initialize Element Class

        Parameters:
        -----------
        cgns_file : str, optional
            Path to CGNS file
        """
        self.GridFileName = GridFileName
        self.Elements = {}
        self.Nodes = None
        self.NodesConnectedToNode = None
        self.CellsConnectedToNode = None
        self.isNodeOnBoundary = None
        self.boundaryNormals = None
        self.BCsAssociatedToNode = None
        self.n_nodes = 0
        self.boundaries = {}
        self.cell_volumes = None
        self.cell_centers = None
        self.boundary_faces = {}
        self.boundary_nodes = {}
        

    def _compute_tetra_properties_Vec(self, nodes, NPoints, alreadyReshaped = False):
        """Compute volume and center for tetrahedron"""
        if not alreadyReshaped:
            print("Compute volume and center for tetrahedrons...")
            nodes = np.reshape(nodes, (int(len(nodes[:, 0])/NPoints), NPoints, 3))
        v1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        v2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        v3 = np.squeeze(nodes[:, 3, :] - nodes[:, 0, :])
        volume = abs(np.sum(v1*np.cross(v2, v3, axis=1), axis=1)) / 6.0
        center = np.mean(nodes, axis=1)
        return volume, center

    def _compute_hexa_properties_Vec(self, nodes, NPoints):
        """Compute volume and center for hexahedron"""
        print("Compute volume and center for hexahedrons...")
        nodes = np.reshape(nodes, (int(len(nodes[:, 0])/NPoints), NPoints, 3))
        center = np.mean(nodes, axis=1)
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
            v, _ = self._compute_tetra_properties_Vec(tet_nodes, 4, alreadyReshaped = True)
            volume += v
        return volume, center

    def _compute_pyramid_properties_Vec(self, nodes, NPoints):
        """Compute volume and center for pyramid"""
        print("Compute volume and center for pyramids...")
        nodes = np.reshape(nodes, (int(len(nodes[:, 0])/NPoints), NPoints, 3))
        base_center = np.mean(nodes[:, :4, :], axis=1)
        apex = np.squeeze(nodes[:, 4, :])
        d1 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 3, :] - nodes[:, 1, :])
        base_area = 0.5 * np.linalg.norm(np.cross(d1, d2, axis=1), axis=1)
        height = np.linalg.norm(apex - base_center, axis=1)
        volume = base_area * height / 3.0
        center = np.mean(nodes, axis=1)
        return volume, center

    def _compute_prism_properties_Vec(self, nodes, NPoints):
        """Compute volume and center for prism (wedge)"""
        print("Compute volume and center for prisms (wedges)...")
        nodes = np.reshape(nodes, (int(len(nodes[:, 0])/NPoints), NPoints, 3))
        v1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        v2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        base_area = 0.5 * np.linalg.norm(np.cross(v1, v2, axis=1), axis=1)
        h1 = np.linalg.norm(np.squeeze(nodes[:, 3, :] - nodes[:, 0, :]), axis=1)
        h2 = np.linalg.norm(np.squeeze(nodes[:, 4, :] - nodes[:, 1, :]), axis=1)
        h3 = np.linalg.norm(np.squeeze(nodes[:, 5, :] - nodes[:, 2, :]), axis=1)
        height = (h1 + h2 + h3) / 3.0
        volume = base_area * height
        center = np.mean(nodes, axis=1)
        return volume, center

    def _compute_normal_quad(self, nodes):
        nodes = np.reshape(nodes, (int(len(nodes[:, 0])/4), 4, 3))
        d1 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 3, :] - nodes[:, 1, :])
        normal = np.cross(d1, d2, axis=1)
        return normal/np.repeat(np.linalg.norm(normal, axis=1)[..., np.newaxis], 3, axis=1)

    def _compute_normal_tria(self, nodes):
        nodes = np.reshape(nodes, (int(len(nodes[:, 0])/3), 3, 3))
        d1 = np.squeeze(nodes[:, 1, :] - nodes[:, 0, :])
        d2 = np.squeeze(nodes[:, 2, :] - nodes[:, 0, :])
        normal = np.cross(d1, d2, axis=1)
        return normal/np.repeat(np.linalg.norm(normal, axis=1)[..., np.newaxis], 3, axis=1)
    

    def _read_cgns_h5py(self, zone_name, BCs):
        """Read CGNS using h5py - NumPy 2.0 compatible"""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py not installed. Run: pip install h5py")
        
        with h5py.File(self.GridFileName, 'r') as f:
            # Navigate CGNS tree structure
            base_name = 'Base'
            base = f[base_name]

            # Find zone
            # print(base.keys())
            # for key in base.keys():
            #     if key not in [' name', ' label', 'ZoneType']:
            #         zone_name = key
            #         break

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
            z = read_coord('CoordinateZ')

            self.Nodes = np.column_stack([x, y, z])
            self.n_nodes = len(self.Nodes)

            isOnBoundary = np.full((len(self.Nodes), 1), False)

            print(f"Loaded {self.n_nodes} nodes")

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

                        # Store elements
                        section_name = key
                        self.Elements[section_name] = {
                            'type': elem_type,
                            'connectivity': connectivity - 1,  # Convert to 0-based
                            'range': (start_idx, end_idx)
                        }

                        print(f"  Section '{section_name}': type={elem_type}, "
                              f"elements={end_idx - start_idx}")
                    except Exception as e:
                        print(f"  Warning: Could not read section '{key}': {e}")
                        continue

            # Read boundary conditions
            
            for bc_name in BCs.keys():
                try:
                    bc = zone[bc_name]
                    # Read boundary face/node list
                    if 'ElementConnectivity' in bc:  # For Pointwise this is actually the element range! PD
                        pl_data = bc['ElementConnectivity']
                        if ' data' in pl_data:
                            node_list = np.array(pl_data[' data']) - 1
                        else:
                            node_list = np.array(pl_data[()]) - 1

                        isOnBoundary[node_list] = True


                        if BCs[bc_name] == 'tri':
                            normals = self._compute_normal_tria(self.Nodes[node_list])
                        if BCs[bc_name] == 'quad':
                            normals = self._compute_normal_quad(self.Nodes[node_list])
                        
                        self.boundary_nodes[bc_name] = {
                                                        'node_list' : node_list,
                                                        'normals': normals,
                                                        'elem_type': BCs[bc_name]
                                                        }
                        self.boundary_faces[bc_name] = bc_name
                        print(f"  Boundary '{bc_name}': {len(node_list)} nodes")

                except Exception as e:
                    print(f"  Warning: Could not read BC '{bc_name}': {e}")
                    continue
            
            self.isNodeOnBoundary = isOnBoundary


        

    def construct_secondaryMeshStructures(self, boundary_conditions):

        print(f"Constructing NodesConnectedToNode structure...")
        self.SetNodesConnectedToNode()

        print(f"Constructing boundary normals structure...")
        self.set_boundary_normals(boundary_conditions)

        print(f"Constructing Cell Volumes and Cell Centers structure...")
        self.compute_geometric_properties()

    def SetNodesConnectedToNode(self):
        # Build node-to-node connectivity from elements
        node_neighbors = defaultdict(set)

        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            # Skip boundary elements
            if elem_type in [2, 3, 5, 7]:
                continue

            nodes_per_elem = self.NODES_PER_ELEMENT.get(elem_type, 0)
            nodesConns_per_elem = self.NODESCONNECTIONS_PER_ELEMENT.get(elem_type, 0)
            if nodes_per_elem == 0:
                continue

            n_elems = len(connectivity) // nodes_per_elem

            # Build connectivity
            for i in range(n_elems):
                if i%10000 == 0:
                    print("Cells analyzed: ", i)
                node_indices = connectivity[i*nodes_per_elem:(i+1)*nodes_per_elem]
                # Each node is connected to all other nodes in the element
                for iNode, iNodeGlobal in enumerate(node_indices):
                    nodeConn = nodesConns_per_elem[iNode]
                    node_neighbors[iNodeGlobal].update(node_indices[nodeConn])
            

        self.NodesConnectedToNode = node_neighbors

    def set_boundary_normals(self, boundary_conditions):
        # Now assign normals to boundary nodes
            boundaryNormal = np.zeros((len(self.Nodes), 3), dtype=float)
            BCsAssociatedToNode = [[] for _ in range(self.n_nodes)]
            for iNode in range(self.n_nodes):
                bnormal = np.zeros((1,3), dtype=float)
                bcs = []
                if self.isNodeOnBoundary[iNode]: # Search only among points on boundaries
                    for bname, bdict in self.boundary_nodes.items():
                        bnodes = bdict['node_list']
                        if iNode in bnodes:
                            bcs = bcs+[bname]
                            bc_name = bname
                            
                            if bname in boundary_conditions:
                                bc_data = boundary_conditions[bname]
                                if bc_data['value'] == 'normal':
                                    # it belongs to a wall 
                                    # reshape the bnodes array to have a clear view of the elements
                                    if bdict['elem_type'] == 'tri':
                                        bnodes = np.reshape(bnodes, (int(len(bnodes)/3), 3))
                                    elif bdict['elem_type'] == 'quad':
                                        bnodes = np.reshape(bnodes, (int(len(bnodes)/4), 4))
                                    
                                    # Then search among all of the elements
                                    boolForElementsWithPoint = (bnodes == iNode).any(axis=1)
                                    normals2Include = bdict['normals'][boolForElementsWithPoint]
                                    bnormal = np.vstack((bnormal, normals2Include))
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

    def compute_geometric_properties(self):
        """
        Compute cell volumes, centers for finite volume method
        """
        print("Computing geometric properties...")

        cell_volumes = np.array([], dtype=float)
        cell_centers = np.array([], dtype=float)

        cell_idx = 0
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']
         
            # Skip boundary elements (2D elements)
            if elem_type in [2, 3, 5, 7]:  # Line, Triangle, Quad
                print(f"  Skipping boundary section '{section_name}' (type={elem_type})")
                continue

            nodes_per_elem = self.NODES_PER_ELEMENT.get(elem_type, 0)
            if nodes_per_elem == 0:
                print(f"  Warning: Unknown element type {elem_type} in '{section_name}'")
                continue

            elem_nodes = self.Nodes[connectivity]

            # Compute volume and center based on element type
            if   elem_type == 10:  # TETRA_4
                volumes, centers = self._compute_tetra_properties_Vec(elem_nodes, nodes_per_elem)
            elif elem_type == 17:  # HEXA_8
                volumes, centers = self._compute_hexa_properties_Vec(elem_nodes, nodes_per_elem)
            elif elem_type == 12:  # PYRA_5
                volumes, centers = self._compute_pyramid_properties_Vec(elem_nodes, nodes_per_elem)
            elif elem_type == 14:  # PENTA_6
                volumes, centers = self._compute_prism_properties_Vec(elem_nodes, nodes_per_elem)
            else:
                # Default approximation for higher-order elements
                print("ERROR! Element type", elem_type, "not recognized!")
                exit(1)

            cell_volumes = np.append(cell_volumes, volumes)
            cell_centers = np.append(cell_centers, centers)

        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers

        print(f"Computed properties for {len(self.cell_volumes)} cells")
        print(f"Volume range: [{self.cell_volumes.min():.6e}, {self.cell_volumes.max():.6e}]")

