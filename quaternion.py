from math import sqrt
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            # Quaternion multiplication
            return Quaternion(
                self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
                self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
                self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            )
        elif isinstance(other, (int, float)):
            # Scalar multiplication
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError("Unsupported operand type for multiplication")
        
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normSquare(self):
        return self.w**2 + self.x**2 + self.y**2 + self.z**2
    
    def norm(self):
        return sqrt(self.normSquare());

    def inverse(self):
        normSquare = self.normSquare()
        if normSquare == 0:
            raise ZeroDivisionError("Cannot invert a quaternion with zero norm")
        return self.conjugate() * (1.0 / normSquare)