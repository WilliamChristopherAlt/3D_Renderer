import numpy as np
import math
import pickle

with open(r'C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.pkl', 'rb') as f:
    objInfo = pickle.load(f)
points = objInfo['points']
noPoints = points.shape[0]
noAdjacents = 5

faceSets = set()

for i, point in enumerate(points):
    distancesI = np.linalg.norm(points - points[i], axis=1)
    adjacentPointsI = np.argsort(distancesI)[1:noAdjacents+1]
    for j in adjacentPointsI:
        distancesJ = np.linalg.norm(points - points[j], axis=1)
        adjacentPointsJ = np.argsort(distancesJ)[1:noAdjacents+1]
        nearByPoints = np.intersect1d(adjacentPointsI, adjacentPointsJ)
        for nearBy in nearByPoints:
            triplet = sorted([i+1, j+1, nearBy+1])
            faceSets.add(tuple(triplet))

print(len(faceSets))

with open(r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.obj", "w") as f:
    f.write('\n'.join(f'v {p[0]} {p[1]} {p[2]}' for p in points))
    f.write('\n\n')
    f.write('\n'.join(f'f {face[0]} {face[1]} {face[2]}' for face in faceSets))

