

import numpy as np

def build_karate_club_graph():
	# All 78 edges are stored in two numpy arrays. One for source endpoints
	# while the other for destination endpoints.
	src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
		10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
		25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
		32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
		33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
	dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
		5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
		24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
		29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
		31, 32])
	# Edges are directional in DGL; Make them bi-directional.
	u = np.concatenate([src, dst])
	v = np.concatenate([dst, src])
	edge_index = np.stack([u,v],0)
	return edge_index
	
