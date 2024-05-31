import numpy as np
import math
import pickle

def isPlane(p1, p2, p3, p4):
    n = np.cross((p1 - p2), (p1 - p3))
    return n @ p1 == n @ p4

def areaTriangle(p1, p2, p3):
    return np.linalg.norm(np.cross((p1 - p2), (p1 - p3))) / 2.0

def areaQuad(p1, p2, p3, p4):
    return (areaTriangle(p1, p2, p3) + areaTriangle(p1, p3, p4)
          + areaTriangle(p1, p2, p4) + areaTriangle(p3, p3, p4)) / 2.0

# with open(r'C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\icosahedron.pkl', 'wb') as f:
#     pickle.dump({'points': points, 'connections': connections}, f)

with open(r'C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\cube.pkl', 'rb') as f:
    objInfo = pickle.load(f)
points = objInfo['points']
noPoints = points.shape[0]
noAdjacents = 3

faceSets = set()
checked = []
squares = []

for i in range(noPoints):
    for j in range(noPoints):
        for k in range(noPoints):
            for l in range(noPoints):
                if len(set([i, j, k, l])) == 4:
                    ordered = sorted([i, j, k, l])
                    if tuple(ordered) not in checked:
                        checked.append(tuple(ordered))
                        if isPlane(points[i], points[j], points[k], points[l]):
                            if (areaQuad(points[i], points[j], points[k], points[l]) == 3.0):
                                squares.append(ordered)

offset = np.array([1, 1, 1])
for square in squares:
    i = square[0]
    j = square[1]
    k = square[2]
    a = points[i]
    b = points[j]
    c = points[k]

    if abs(np.linalg.norm(a - b) - 2) < 0.001:
        if (abs(np.linalg.norm(b - c) - 2) < 0.001):
            ordered = sorted([i, j, square[2]])
            faceSets.add(tuple(np.array(ordered) + offset))
            ordered = sorted([i, square[2], square[3]])
            faceSets.add(tuple(np.array(ordered) + offset))
        else:
            ordered = sorted([i, j, square[2]])
            faceSets.add(tuple(np.array(ordered) + offset))
            ordered = sorted([j, square[2], square[3]])
            faceSets.add(tuple(np.array(ordered) + offset))
    else:
        print(False)
        ordered = sorted([i, j, square[2]])
        faceSets.add(tuple(np.array(ordered) + offset))
        ordered = sorted([i, j, square[3]])
        faceSets.add(tuple(np.array(ordered) + offset))


print(len(faceSets))
for face in faceSets:
    print(face)

with open(r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\cube.obj", "w") as f:
    f.write('\n'.join(f'v {p[0]} {p[1]} {p[2]}' for p in points))
    f.write('\n\n')
    f.write('\n'.join(f'f {face[0]} {face[1]} {face[2]}' for face in faceSets))

