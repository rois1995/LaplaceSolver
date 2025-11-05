import numpy as np

class ElementsUtilities:
        
    def __init__(self, Logger):
        self.Logger = Logger

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
        5: [[0, 5], [1, 3], [2, 4]],    # TRI_3
        7: [[0, 7], [1, 4], [2, 5], [3, 6]],    # QUAD_4
        # 7: [[0, 3], [0, 1], [1, 2], [2, 3]],    # QUAD_4

        10: [[0, 2, 3], [0, 1, 4], [1, 2, 5], [3, 4, 5]],   # TETRA_4

        12: [[0, 4, 11], [1, 5, 8], [2, 6, 9], [3, 7, 10], [12, 13, 14, 15]],   # PYRA_5

        14: [[0, 2, 6], [0, 1, 7], [1, 2, 8], 
             [3, 5, 6], [3, 4, 7], [4, 5, 8]],   # PENTA_6

        17: [[0, 11, 15], [1, 8, 12], [2, 9, 13], [3, 10, 14], 
             [4, 19, 23], [5, 16, 20], [6, 17, 21], [7, 18, 22]],   # HEXA_8  As per CGNS standard
    }

    # Edges shared by the same Node per element for each type
    SIGN_EDGESOFELEMENT_PER_ELEMENT = {
        5: [1, 1, 1, -1, -1, -1],    # TRI_3
        7: [1, 1, 1, 1, -1, -1, -1, -1],    # QUAD_4

        10: [],   # TETRA_4

        12: [],   # PYRA_5

        14: [],   # PENTA_6

        17: [],   # HEXA_8  As per CGNS standard
    }

    # Edges as NodeI-NodeJ in local indices per element for each type
    EDGES_PER_ELEMENT = {
        5: [[0, 1], [1, 2], [2, 0],
            [1, 0], [2, 1], [0, 2]],    # TRI_3
        # 7: [[0, 1], [1, 2], [2, 3], [3, 0]],    # QUAD_4
        7: [[0, 1], [1, 2], [2, 3], [3, 0],
            [1, 0], [2, 1], [3, 2], [0, 3]],    # QUAD_4

        10: [[0, 1], [1, 2], [2, 0],
             [0, 3], [1, 3], [2, 3],
             
             [1, 0], [2, 1], [0, 2],
             [3, 0], [3, 1], [3, 2]
             ],   # TETRA_4

        12: [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 4], [2, 4], [3, 4],
             
             [1, 0], [2, 1], [3, 2], [0, 3],
             [4, 0], [4, 1], [4, 2], [4, 3]
             ],   # PYRA_5

        14: [[0, 1], [1, 2], [2, 0],
             [3, 4], [4, 5], [5, 3],
             [0, 3], [1, 4], [2, 5],
             
             [1, 0], [2, 1], [0, 2],
             [4, 3], [5, 4], [3, 5],
             [3, 0], [4, 1], [5, 2]
             ],   # PENTA_6

        17: [[0, 1], [1, 2], [2, 3], [3, 0], 
             [4, 5], [5, 6], [6, 7], [7, 4], 
             [1, 5], [2, 6], [3, 7], [0, 4],
             
             [1, 0], [2, 1], [3, 2], [0, 3], 
             [5, 4], [6, 5], [7, 6], [4, 7], 
             [5, 1], [6, 2], [7, 3], [4, 0]
             ],   # HEXA_8
    }



    # Faces (as collection of nodes) associated to each element for each type (3D Only)
    FACES_PER_ELEMENT = {
        5: [],    # TRI_3
        7: [],    # QUAD_4

        10: [[0, 2, 1], [0, 1, 3], [2, 0, 3], [1, 2, 3]],   # TETRA_4

        12: [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],   # PYRA_5

        14: [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5], [0, 2, 1], [4, 5, 3]],   # PENTA_6

        17: [[0, 3, 2, 1], [0, 1, 5, 4], [0, 4, 7, 3], [1, 2, 6, 5],
             [4, 5, 6, 7], [2, 3, 7, 6]],   # HEXA_8
    }


    # Mask for which face is associated to an edge of the element for each type (3D Only)
    FACESOFEDGES_PER_ELEMENT = {
        5: [],    # TRI_3
        7: [],    # QUAD_4
        10: [[0, 1], [0, 3], [0, 2],
             [1, 2], [1, 3], [2, 3],
             
             [0, 1], [0, 3], [0, 2],
             [1, 2], [1, 3], [2, 3]],   # TETRA_4

        12: [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 4], [1, 2], [2, 3], [3, 4],
             
             [0, 1], [0, 2], [0, 3], [0, 4],
             [1, 4], [1, 2], [2, 3], [3, 4]
             ],   # PYRA_5

        14: [[0, 3], [1, 3], [2, 3],
             [0, 4], [1, 4], [2, 4],
             [0, 2], [0, 1], [1, 2],
             
             [0, 3], [1, 3], [2, 3],
             [0, 4], [1, 4], [2, 4],
             [0, 2], [0, 1], [1, 2]
             ],   # PENTA_6

        17: [[0, 1], [0, 3], [0, 5], [0, 2], 
             [4, 1], [4, 3], [4, 5], [2, 4], 
             [1, 3], [3, 5], [2, 5], [1, 2],
             
             [0, 1], [0, 3], [0, 5], [0, 2], 
             [4, 1], [4, 3], [4, 5], [2, 4], 
             [1, 3], [3, 5], [2, 5], [1, 2]],   # HEXA_8
    }


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

        CVAreas = np.zeros((nodes[:, :, 0].shape), dtype=np.float64)
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
        
        CVAreas = np.zeros((nodes[:, :, 0].shape), dtype=np.float64)
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
    