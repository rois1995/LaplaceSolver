import numpy as np
from collections import defaultdict
import warnings
import time
import sys
from ElementsUtilities import ElementsUtilities
from numba.typed import Dict as numbaDict
from numba import types as numbaTypes
from Parameters import BC_TYPE_MAP, EXACT_SOLUTION_MAP

class MeshClass:

    
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
        self.NumOfNodesConnectedToNode = None
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
        self.ElementsUtilities = ElementsUtilities(Logger)
        

    

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
                        nPointsPerElement = self.ElementsUtilities.NODES_PER_ELEMENT[elem_type]
                        connectivity = np.reshape(connectivity-1, (int(len(connectivity)/nPointsPerElement), nPointsPerElement))

                        # Store elements
                        section_name = self.ElementsUtilities.ELEMENT_NAMES_FROM_TYPES[str(elem_type)]
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

                        elem_type = self.ElementsUtilities.ELEMENT_TYPES_FROM_NAMES[BCs[bc_name]['Elem_type']]
                        nPointsPerElement = self.ElementsUtilities.NODES_PER_ELEMENT[elem_type]
                        connectivity = np.reshape(connectivity, (int(len(connectivity)/nPointsPerElement), nPointsPerElement))
                        
                        if BCs[bc_name]['Elem_type'] == 'line':
                            normals = self.ElementsUtilities._compute_normal_line(self.Nodes[connectivity])
                            areas = self.ElementsUtilities._compute_line_length(self.Nodes[connectivity])
                            CVAreas = self.ElementsUtilities._compute_CV_contrib_line(self.Nodes[connectivity])
                        elif BCs[bc_name]['Elem_type'] == 'tri':
                            normals = self.ElementsUtilities._compute_normal_tria(self.Nodes[connectivity])
                            areas = self.ElementsUtilities._compute_tria_area(self.Nodes[connectivity])
                            CVAreas = self.ElementsUtilities._compute_CV_contrib_tria(self.Nodes[connectivity])
                        elif BCs[bc_name]['Elem_type'] == 'quad':
                            normals = self.ElementsUtilities._compute_normal_quad(self.Nodes[connectivity])
                            areas = self.ElementsUtilities._compute_quad_area(self.Nodes[connectivity])
                            CVAreas = self.ElementsUtilities._compute_CV_contrib_quad(self.Nodes[connectivity])
                        
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
        
        startTime = time.time()
        self.Logger.info(f"Constructing NodesConnectedToNode structure...")
        self.SetNodesConnectedToNode()
        self.Logger.info(f"Finished construction of NodesConnectedToNode structure. Elapsed time {time.time()-startTime} s.")

        startTime = time.time()
        self.Logger.info(f"Constructing boundary structure...")
        self.set_boundary_Variables(boundary_conditions)
        self.Logger.info(f"Finished construction of boundary structure. Elapsed time {time.time()-startTime} s.")

        startTime = time.time()
        self.Logger.info(f"Constructing Cell Centers structure...")
        self.compute_geometric_properties()
        self.Logger.info(f"Finished construction of Cell Centers structure. Elapsed time {time.time()-startTime} s.")

    def SetNodesConnectedToNode(self):
        # Build node-to-node connectivity from elements
        node_neighbors = [set() for i in range(self.n_nodes)]
        node_cells = [[] for _ in range(self.n_nodes)]

        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            nodes_per_elem = self.ElementsUtilities.NODES_PER_ELEMENT.get(elem_type, 0)
            nodesConns_per_elem = self.ElementsUtilities.NODESCONNECTIONS_PER_ELEMENT.get(elem_type, 0)
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
            


        NNeighbors = np.zeros((self.n_nodes,),dtype=int)
        for iNode in range(self.n_nodes):
            NNeighbors[iNode] = len(node_neighbors[iNode])
        
        self.NumOfNodesConnectedToNode = NNeighbors
        self.NodesConnectedToNode = node_neighbors
        self.CellsConnectedToNode = node_cells

    def set_boundary_Variables(self, boundary_conditions):
        # Now assign normals to boundary nodes
            boundaryNormal = np.zeros((len(self.Nodes), 3), dtype=np.float64)
            HowManyNormals = np.zeros((len(self.Nodes), ), dtype=np.int32)
            boundaryCVArea = np.zeros((len(self.Nodes), ), dtype=np.float64)
            BCsAssociatedToNode = np.empty((len(self.Nodes), ), dtype='U40')

            # BCsAssociatedToNode = [[] for _ in range(self.n_nodes)]

            for bname, bdict in self.boundary_nodes.items():
                bnodes = bdict['connectivity']
                nPointsPerElem = len(bnodes[0, :])
                normals = bdict['normals']
                areas = bdict['BoundaryCVArea']

                np.add.at(boundaryNormal, bnodes.flatten(), np.repeat(normals, nPointsPerElem, axis=0))
                np.add.at(HowManyNormals, bnodes.flatten(), 1)
                np.add.at(boundaryCVArea, bnodes.flatten(), np.repeat(areas[: ,0], nPointsPerElem, axis=0))

                mask = np.logical_not(HowManyNormals== 0)
                boundaryNormal[mask, :] /= np.repeat(HowManyNormals[mask, np.newaxis], 3, axis=1)

            for iBoundNode in np.where(self.isNodeOnBoundary)[0]:
                for bname, bdict in self.boundary_nodes.items():
                    bnodes = bdict['connectivity']
                    if iBoundNode in bnodes:
                        BCsAssociatedToNode[iBoundNode] = bname  # I don't really care about how many are there,
                                                                 # since for this problem all nodes belonging
                                                                 # to the same surface will have the same BC
                        
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
         

            nodes_per_elem = self.ElementsUtilities.NODES_PER_ELEMENT.get(elem_type, 0)
            if nodes_per_elem == 0:
                self.Logger.info(f"  Warning: Unknown element type {elem_type} in '{section_name}'")
                continue

            elem_nodes = self.Nodes[connectivity]

            # Pretty sure I just need the centroid
            centers = np.mean(elem_nodes, axis=1)

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
        ControlFaceDictPerEdge = [{} for i in range(self.n_nodes)]
        ControlFaceNormalDictPerEdge = [{} for i in range(self.n_nodes)]
        for iNode in range(self.n_nodes):

            neighbors = list(self.NodesConnectedToNode[iNode])
            for neigh in neighbors:
                ControlFaceDictPerEdge[iNode][str(neigh)] = 0.0
                ControlFaceNormalDictPerEdge[iNode][str(neigh)] = np.array([0.0, 0.0, 0.0])

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")


        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")

        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT.get(elem_type, 0)

            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            edgeCentroids = np.mean(ElementNodes[:, edges_per_elem, :], axis=2)

            self.Elements[section_name]["edgeCentroids"] = edgeCentroids
            
            SectionalAreaOfEdges = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem)), dtype=float)
            LineNormalsOfEdgeFaces = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem), 3), dtype=float)

            lines = np.zeros((len(elem_data["connectivity"][:, 0]), 2, 3), dtype=float)

            lines[:, 1, :] = elem_data["CellCenters"]

            signOf_EdgesOfElements = self.ElementsUtilities.SIGN_EDGESOFELEMENT_PER_ELEMENT.get(elem_type, 0)

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            for iEdge, edge in enumerate(edges_per_elem):

                lines[:, 0, :] = edgeCentroids[:, iEdge]

                LineNormalsOfEdgeFaces[:, iEdge, :] = self.ElementsUtilities._compute_normal_line(lines)
                SectionalAreaOfEdges[:, iEdge] = self.ElementsUtilities._compute_line_length(lines, toPrint=False)

                signOfEdge = signOf_EdgesOfElements[iEdge]
                
                for iElem in range(len(connectivity[:, 0])):
                    PointStart = connectivity[iElem, edge[0]]
                    PointEnd = connectivity[iElem, edge[1]]
                    ControlFaceDictPerEdge[PointStart][str(PointEnd)] += SectionalAreaOfEdges[iElem, iEdge]
                    ControlFaceNormalDictPerEdge[PointStart][str(PointEnd)] += LineNormalsOfEdgeFaces[iElem, iEdge, :]*signOfEdge*SectionalAreaOfEdges[iElem, iEdge]

            edgesOfPoints = self.ElementsUtilities.EDGESOFPOINTS_PER_ELEMENT.get(elem_type, 0)
            trias = np.zeros((len(elem_data["connectivity"][:, 0]), 3, 3), dtype=float)
            trias[:, 0, :] = elem_data["CellCenters"]

            for iPoint in range(self.ElementsUtilities.NODES_PER_ELEMENT.get(elem_type, 0)):

                pointsGlobalIndices = elem_data["connectivity"][:, iPoint]

                trias[:, 1, :] = self.Nodes[pointsGlobalIndices, :]

                # Now I compute the areas of each edge midpoint
                # Cycle on each edge
                for iEdge, edge in enumerate(edgesOfPoints[iPoint]):

                    trias[:, 2, :] = edgeCentroids[:, edge]

                    controlVolumeContrib = self.ElementsUtilities._compute_tria_area(trias, toPrint=False)
                    # I can already add it to the global index
                    np.add.at(ControlVolumesPerNode, pointsGlobalIndices, controlVolumeContrib)
        
        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Collecting the edge face area from every element")

        NodesOfNodes = self.NodesConnectedToNode
        maxNNeighs = max(self.NumOfNodesConnectedToNode)
        NumbaControlFaceNormalDictPerEdge = np.full((len(NodesOfNodes), maxNNeighs, 3), -1, dtype=np.float64)
        NumbaControlFaceDictPerEdge = np.full((len(NodesOfNodes), maxNNeighs), -1, dtype=np.float64)

        for iNode in range(self.n_nodes):
            for iNeigh, neigh in enumerate(NodesOfNodes[iNode]):
                key = str(neigh)
                NumbaControlFaceDictPerEdge[iNode, iNeigh] = ControlFaceDictPerEdge[iNode][key]
                NumbaControlFaceNormalDictPerEdge[iNode, iNeigh] = ControlFaceNormalDictPerEdge[iNode][key]/ControlFaceDictPerEdge[iNode][key]

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        
        self.ControlFaceDictPerEdge = NumbaControlFaceDictPerEdge
        self.ControlFaceNormalDictPerEdge = NumbaControlFaceNormalDictPerEdge
        self.ControlVolumesPerNode = ControlVolumesPerNode

    def build_DualControlVolumes_3D(self):

        # Let's start from computing the control volumes associated 
        # to each node and their elements.

        start = time.time()
        self.Logger.info("Initializing ControlVolumes per Node dictionary...")


        # Construct a dictionary of edges for each point that will have the area associated to it
        ControlVolumesPerNode = np.zeros((self.n_nodes, ), dtype=float)
        ControlFaceDictPerEdge = [{} for i in range(self.n_nodes)]
        ControlFaceNormalDictPerEdge = [{} for i in range(self.n_nodes)]
        for iNode in range(self.n_nodes):

            neighbors = list(self.NodesConnectedToNode[iNode])
            for neigh in neighbors:
                ControlFaceDictPerEdge[iNode][str(neigh)] = 0.0
                ControlFaceNormalDictPerEdge[iNode][str(neigh)] = np.array([0.0, 0.0, 0.0])

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")


        start = time.time()
        self.Logger.info("Computing the edges (and faces in 3D) centroids...")
        self.Logger.info("Computing Control Volume and face area between node i and j...")

        # Cycle on each element and pre-compute edges mid-points and faces centers if 3D
        for section_name, elem_data in self.Elements.items():
            elem_type = elem_data['type']
            connectivity = elem_data['connectivity']

            edges_per_elem = self.ElementsUtilities.EDGES_PER_ELEMENT.get(elem_type, 0)

            # Compute the median of all the edges
            ElementNodes = self.Nodes[connectivity]

            edgeCentroids = np.mean(ElementNodes[:, edges_per_elem, :], axis=2)

            self.Elements[section_name]["edgeCentroids"] = edgeCentroids

            faces_per_elem = self.ElementsUtilities.FACES_PER_ELEMENT.get(elem_type, 0)
            faceCentroids = np.zeros((len(connectivity[:, 0]), len(faces_per_elem), 3), dtype=float)
            for iFace, face in enumerate(faces_per_elem):
                faceCentroids[:, iFace, :] = np.mean(ElementNodes[:, face, :], axis=1)
            self.Elements[section_name]["FaceCentroids"] = faceCentroids

            facesOfEdges = self.ElementsUtilities.FACESOFEDGES_PER_ELEMENT.get(elem_type, 0)

            SectionalAreaOfEdges = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem)), dtype=float)
            NormalsOfEdgeFaces = np.zeros((len(elem_data["connectivity"][:, 0]), len(edges_per_elem), 3), dtype=float)

            trias = np.zeros((len(elem_data["connectivity"][:, 0]), 3, 3), dtype=float)

            trias[:, 0, :] = elem_data["CellCenters"]

            # Now I compute the areas of each edge midpoint
            # Cycle on each edge
            for iEdge, edge in enumerate(edges_per_elem):

                trias[:, 2, :] = edgeCentroids[:, iEdge]

                edgeName = ''.join(map(str, edge))

                for face in facesOfEdges[iEdge]:
                                            
                    # Compose tria and compute area contribution
                    trias[:, 1, :] = faceCentroids[:, face, :]
                    faceAreaContrib = self.ElementsUtilities._compute_tria_area(trias, toPrint=False)

                    faceName = ''.join(map(str, faces_per_elem[face] + [faces_per_elem[face][0]]))
                    signOfFace = -1
                    if edgeName in faceName:
                        signOfFace = 1
                    
                    normal = self.ElementsUtilities._compute_normal_tria(trias)
                    NormalsOfEdgeFaces[:, iEdge, :] += normal * signOfFace * np.repeat(faceAreaContrib[..., np.newaxis], 3, axis=1)
                    SectionalAreaOfEdges[:, iEdge] += faceAreaContrib
                
                for iElem in range(len(connectivity[:, 0])):
                    PointStart = connectivity[iElem, edge[0]]
                    PointEnd = connectivity[iElem, edge[1]]
                    ControlFaceDictPerEdge[PointStart][str(PointEnd)] += SectionalAreaOfEdges[iElem, iEdge]
                    ControlFaceNormalDictPerEdge[PointStart][str(PointEnd)] += NormalsOfEdgeFaces[iElem, iEdge, :]

            edgesOfPoints = self.ElementsUtilities.EDGESOFPOINTS_PER_ELEMENT.get(elem_type, 0)
            tetras = np.zeros((len(elem_data["connectivity"][:, 0]), 4, 3), dtype=float)
            tetras[:, 0, :] = elem_data["CellCenters"]

            for iPoint in range(self.ElementsUtilities.NODES_PER_ELEMENT.get(elem_type, 0)):

                pointsGlobalIndices = elem_data["connectivity"][:, iPoint]

                tetras[:, 1, :] = self.Nodes[pointsGlobalIndices, :]

                # Now I compute the areas of each edge midpoint
                # Cycle on each edge
                for iEdge, edge in enumerate(edgesOfPoints[iPoint]):

                    tetras[:, 2, :] = edgeCentroids[:, edge]

                    for face in facesOfEdges[edge]:
                                                
                        # Compose tetra and compute the volume
                        tetras[:, 3, :] = faceCentroids[:, face, :]
                        controlVolumeContrib = self.ElementsUtilities._compute_tetra_volume(tetras, toPrint=False)

                        # I can already add it to the global index
                        np.add.at(ControlVolumesPerNode, pointsGlobalIndices, controlVolumeContrib)
        
        # exit(1)

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")

        start = time.time()
        self.Logger.info("Collecting the edge face area from every element")

        NodesOfNodes = self.NodesConnectedToNode
        maxNNeighs = max(self.NumOfNodesConnectedToNode)
        NumbaControlFaceNormalDictPerEdge = np.full((len(NodesOfNodes), maxNNeighs, 3), -1, dtype=np.float64)
        NumbaControlFaceDictPerEdge = np.full((len(NodesOfNodes), maxNNeighs), -1, dtype=np.float64)

        for iNode in range(self.n_nodes):
            for iNeigh, neigh in enumerate(NodesOfNodes[iNode]):
                key = str(neigh)
                NumbaControlFaceDictPerEdge[iNode, iNeigh] = ControlFaceDictPerEdge[iNode][key]
                NumbaControlFaceNormalDictPerEdge[iNode, iNeigh] = ControlFaceNormalDictPerEdge[iNode][key]/ControlFaceDictPerEdge[iNode][key]

        self.Logger.info(f"Done! Elapsed time {str(time.time()-start)} s")
        
        self.ControlFaceDictPerEdge = NumbaControlFaceDictPerEdge
        self.ControlFaceNormalDictPerEdge = NumbaControlFaceNormalDictPerEdge
        self.ControlVolumesPerNode = ControlVolumesPerNode

        








        

                
                
                
