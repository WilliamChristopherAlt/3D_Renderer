import numpy as np
class WavefrontObject:
    def __init__(self, objFile):
        with open(objFile, 'r') as f:
            verticesCount = sum(1 for line in f if line.startswith('v'))
            f.seek(0)
            facesCount = sum(1 for line in f if line.startswith('f'))
            f.seek(0)

            self.vertices = np.zeros((verticesCount, 3), dtype=float)
            self.faces = np.zeros((facesCount, 3), dtype=int)

            verticeIDX = 0
            faceIDX = 0

            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    self.vertices[verticeIDX] = np.array(list(map(float, parts[1:])))
                    verticeIDX += 1
                elif parts[0] == 'f':
                    # Face line: "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
                    infoLength = len(parts[1].split('/'))
                    if infoLength == 1:
                        self.faces[faceIDX] = np.array(list(map(int, parts[1:])))
                    elif infoLength == 2:
                        vertices = [int(part.split("/")[0]) for part in parts[1:]]
                        self.faces[faceIDX] = np.array(vertices)
                    else:
                        vertices = [int(part.split("/")[0]) for part in parts[1:]]
                        self.faces[faceIDX] = np.array(vertices)

                    faceIDX += 1

        self.faces -= 1 # Some how they start index from 1, which is obnoxious
