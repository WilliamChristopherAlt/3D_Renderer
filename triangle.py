import numpy as np
from math import floor

def drawTriangle(x0, y0, x1, y1, x2, y2):
    minX = floor(min(x0, x1, x2))
    minY = floor(min(y0, y1, y2))
    maxX = floor(max(x0, x1, x2))
    maxY = floor(max(y0, y1, y2))

    xs = []
    ys = []

    s1 = y2 - y0

    if s1 == 0:
        s1 = 0.00001

    s2 = x2 - x0
    s3 = y1 - y0
    s4 = (s3 * s2 - (x1 - x0) * s1)

    if s4 == 0.0:
        s4 = 0.0001

    for x in range(minX, maxX+1):
        for y in range(minY, maxY+1):
            try:
                s5 = y - y0
                w1 = (x0 * s1 + s5 * s2 - x * s1) / s4
                w2 = (s5 - w1 * s3) / s1

                if (w1 >= 0 and w2 >= 0 and w1 + w2 <= 1):
                    xs.append(x)
                    ys.append(y)
            except:
                print("Zero division occured when drawing triangle")
                continue
    return np.array(xs), np.array(ys)

# import numpy as np
# from matplotlib import pyplot as plt
# grid = np.zeros((100, 100), dtype=bool)
# xs, ys = drawTriangle(10, 10, 10, 80, 80, 10)
# grid[ys, xs] = True
# plt.imshow(grid, cmap='grey')
# plt.show()