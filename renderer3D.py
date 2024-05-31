# PREREQUISITE: 
# + Quaternion to represent 3D rotation
# + Bresenham Algorithm
# + Projection from 3D to 2D plane
# + 2D linear equation solver

import numpy as np
from matplotlib import pyplot as plt
from math import floor, sin, cos
from matplotlib.animation import FuncAnimation
from matplotlib import animation

from quaternion import Quaternion
from wavefront import WavefrontObject
from triangle import drawTriangle

frames = 500
gridXSize = 256
rotateBy = 0.03
zoom = 1.0
dt = 50 # Milliseconds

initialOffset = np.array([0, 0, 0], dtype=float)
initialRotation = 1.5*np.pi/3

# axisList = np.array([[0, 1, 0], [0, 1, 1]], dtype=float)
axisList = np.array([[0, 1, 0]], dtype=float)
axisList = axisList / np.linalg.norm(axisList, axis=1)[:, np.newaxis]

# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\miku.obj"
# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\cirno.obj"
objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\rin.obj"
# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\Hatsune_Miku_Tentacles.obj"
# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\Suzanne_Blender_Monkey.obj"
# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\cube.obj"
# objFile = r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\Data\icosahedron.obj"

def solve2D(a, b, c, d, e, f):
    deno = a*e - d*b
    try:
        return (c*e - b*f) / deno, (a*f - d*c) / deno
    except ZeroDivisionError:
        return 0.0, 0.0

def animate(i, graph):
    print(f'{frames-i} frames remained')
    graph.rotate()
    plt.cla()
    graph.draw()
    
def create_animation(graph, dt):
    fig_animation = plt.figure()
    ax_animation = plt.axes()
    fig_animation.patch.set_facecolor('xkcd:black')
    ax_animation.set_facecolor([0.1, 0.15, 0.15])
    ax_animation.tick_params(axis='x', which='both', bottom=False,
                             top=False, labelbottom=False)
    ax_animation.tick_params(axis='y', which='both', right=False,
                             left=False, labelleft=False)
    plt.tight_layout()

    ani = FuncAnimation(fig_animation, animate, fargs=(graph, ), frames=frames , interval=dt)

    return ani

class Graph:
    def __init__(self, gridXSize, axisList, rotateBy, wfObject, initialAxis, initialRotation, initialOffset, zoom):
        self.vertices = wfObject.vertices

        centerOfMass = np.average(self.vertices, axis=0)
        self.vertices -= 2.3*centerOfMass

        avgDisToOrigin = np.average(np.linalg.norm(self.vertices, axis=1))
        self.vertices /= avgDisToOrigin

        for vertex in self.vertices:
            vertex += initialOffset
            rotate(vertex, initialAxis, initialRotation)

        self.faces = wfObject.faces
        self.noPoints = self.vertices.shape[0]
        self.noFaces = self.faces.shape[0] # ha ha get it anyone?

        self.minX = np.min(self.vertices[:, 0])
        self.minY = np.min(self.vertices[:, 1])
        self.maxX = np.max(self.vertices[:, 0])
        self.maxY = np.max(self.vertices[:, 1])

        if self.minX >= self.maxX or self.minY >= self.maxY:
            raise ValueError   

        self.lengthX = self.maxX - self.minX
        self.lengthY = self.maxY - self.minY

        self.gridXSize = gridXSize
        self.gridYSize = floor(gridXSize * self.lengthY / self.lengthX)

        self.scaleWorldX = self.lengthX / self.gridXSize
        self.scaleWorldY = self.lengthY / self.gridYSize

        self.scalePixelX = 1.0 / self.scaleWorldX
        self.scalePixelY = 1.0 / self.scaleWorldY

        self.vertices *= zoom

        self.axisList = axisList
        self.rotateBy = rotateBy

        self.grid = np.zeros((self.gridYSize, self.gridXSize), dtype=float)
        self.depthGrid= np.full((self.gridYSize, self.gridXSize), float('inf'), dtype=float)

        self.facesNormal = np.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                                self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]], axis=1)
        self.facesNormal /= np.linalg.norm(self.facesNormal, axis=1)[:, np.newaxis]

        self.rotateVectorized = np.vectorize(rotate)

    def rotate(self):
        # print(self.facesNormal)
        for vertex in self.vertices:
            for axis in self.axisList:
                rotate(vertex, axis, self.rotateBy)

        self.facesNormal = np.cross(self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]],
                                self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 0]], axis=1)
        self.facesNormal /= np.linalg.norm(self.facesNormal, axis=1)[:, np.newaxis]

    def draw(self):
        self.grid.fill(0.0)
        self.depthGrid.fill(float('inf'))
        
        for idx, face in enumerate(self.faces):     
            n = self.facesNormal[idx]
            if (n[2] == 0):
                continue

            i = face[0]
            j = face[1]
            k = face[2] 

            xa = self.vertices[i][0]
            ya = self.vertices[i][1]
            xb = self.vertices[j][0]
            yb = self.vertices[j][1]
            xc = self.vertices[k][0]
            yc = self.vertices[k][1]

            x0 = (xa - self.minX) * self.scalePixelX
            y0 = (-ya - self.minY) * self.scalePixelY
            x1 = (xb - self.minX) * self.scalePixelX
            y1 = (-yb - self.minY) * self.scalePixelY
            x2 = (xc - self.minX) * self.scalePixelX
            y2 = (-yc - self.minY) * self.scalePixelY

            xs, ys = drawTriangle(x0, y0, x1, y1, x2, y2)
            valid_indices = (xs >= 0) & (xs < self.gridXSize) & (ys >= 0) & (ys < self.gridYSize)
            xs = xs[valid_indices]
            ys = ys[valid_indices]

            d = n @ self.vertices[face[0]]

            # Project point (x, y, 0) back to the triangle to find z depth
            for pixX, pixY in zip(xs, ys):
                x = pixX * self.scaleWorldX + self.minX
                y = -pixY * self.scaleWorldY - self.minY

                zDepth = (d - n[0] * x - n[1] * y) / n[2]

                if self.depthGrid[pixY, pixX] > zDepth:
                    self.depthGrid[pixY, pixX] = zDepth
                    self.grid[pixY, pixX] = n[0] - 0.3*n[2]
        plt.imshow(self.grid, cmap='grey', vmin=0, vmax=1)

# Rotate p around axis(normalize) by angle
def rotate(p, axis, angle):
    quaternion = Quaternion(0, p[0], p[1], p[2])

    # Construct the quaternion for rotation
    sinVal = sin(angle / 2.0)
    q = Quaternion(cos(angle / 2.0), sinVal * axis[0], sinVal * axis[1], sinVal * axis[2])

    # q = Quaternion(cos(angle / 2.0), 0.0, 0.0, 0.0) + axis * sin(angle / 2.0)
    rotated = q * quaternion * q.inverse()
    p[0] = rotated.x
    p[1] = rotated.y
    p[2] = rotated.z

# Project p to plane with normal vector n and passes through p0
def project(p, n, p0):
    return p - n * (n @ (p0 - p) / (-(n**2).sum()))

def main(): 
    graph = Graph(gridXSize, axisList, rotateBy, WavefrontObject(objFile), np.array([0, 1, 0], dtype=float), initialRotation, initialOffset, zoom)

    ani = create_animation(graph, dt)
    # plt.show()
    FFwriter = animation.FFMpegWriter(fps=20)
    ani.save(r"C:\Users\PC\Desktop\Programming\Python\ownProjects\3D Renderer\rin.mp4", writer = FFwriter, dpi=200)

main()