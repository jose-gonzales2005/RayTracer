from math import sqrt, pow

def point (x, y, z):
	return Tuple(x, y, z, 1)

def vector(x, y, z):
	return Tuple(x, y, z, 0)






class Tuple:

	def __init__(self, x1, y1, z1, w1):
		self.x = float(x1)
		self.y = float(y1)
		self.z = float(z1)
		self.w = float(w1)

	def __str__(self):
		return "Tuple[x = %f, y = %f, z = %f, w = %f" % (self.x, self.y, self.z, self.w)

	def __repr__(self):
		return self.__str__()

	def add(self, other):
		return Tuple(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

	def sub(self, other):
		return Tuple(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

	def negate(self):
		return Tuple(-self.x, -self.y, -self.z, -self.w)

	def mul(self, m):
		return Tuple(self.x*m, self.y*m, self.z*m, self.w*m)

	def div(self, d):
		return Tuple(self.x/d, self.y/d, self.z/d, self.w/d)

	def magnitude(self):
		return sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)

	def normalize(self):
		m = self.magnitude()
		return Tuple(self.x/m, self.y/m, self.z/m, self.w/m)

	def dot(self, other):
		return self.x*other.x + self.y*other.y + self.z*other.z +self.w*other.w #dot and cross are dot and cross product, include in vid

	def cross(self, other):
		return vector(
			self.y*other.z - self.z*other.y, 
			self.z*other.x - self.x*other.z, 
			self.x*other.y - self.y*other.x)


	def __eq__(self, other):
		if isinstance(other, Tuple):
			return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w
		return false 

	def __ne__(self, other):
		return not self.__eq__(other)

	def transform(self, m):
		return m.mul_tuple(self)

	def reflect(self, normal):
		return self.sub(normal.mul(2 * self.dot(normal)))

def clamp(c):
	if c < 0:
		return 0
	elif c > 255:
		return 255
	else:
		return c




class Color:
	def __init__(self, r, g, b):
		self.red = float(r)
		self.green = float(g)
		self.blue = float(b)

	def __repr__(self):
		return "Color [r:{} g:{} b:{}]".format(self.red, self.green, self.blue)

	def add(self, other):
		return Color(self.red + other.red, self.green + other.green, self.blue + other.blue)

	def sub(self, other):
		return Color(self.red + other.red, self.green + other.green, self.blue + other.blue)

	def mul_scalar(self, m):
		return Color(self.red*m, self.green*m, self.blue*m)

	def mul(self, other):
		return Color(self.red*other.red, self.green*other.green, self.blue*other.blue)

	def to_ppm(self):
		return "{} {} {} ".format(
			clamp(int(self.red*255)), 
			clamp(int(self.green*255)),
			clamp(int(self.blue*255))
				)



class Canvas:

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.data = [Color(0, 0, 0)] * (width*height)

	def write_pixel(self, x, y, color):
		self.data[y*self.width + x]	= color

	def pixel_at(self, x, y):
		return self.data[y*self.width + x]

	def fill(self, color):
		for y in range(self.height):
			for x in range(self.width):
				write_pixel(x, y, color)

	def save(self, filename):
		f = open(filename, 'w')
		f.write("P3\n")
		f.write("{} {}\n".format(self.width, self.height))
		f.write("255\n")
		for y in range(self.height):
			for x in range(self.width):
				f.write(self.pixel_at(x, y).to_ppm())
			f.write("\n")
		f.close()

class Ray:

	def __init__(self, o, d):
		self.origin = o
		self.direction = d

	def __repr__(self):
		return "Ray[origin: {}  direction: {}]".format(self.origin, self.direction)

	def position(self, t):
		return self.origin.add(self.direction.mul(t))

	def transform(self, m):
		return Ray(
			self.origin.transform(m),
			self.direction.transform(m)
		)

class Intersection:

	def __init__(self, shape, t):
		self.t = t
		self.shape = shape

	def __eq__(self, other):
		if isinstance(other, Intersection):
			return self.shape == other.shape and self.t == other.t
		return false

	def __ne__(self, other):
		return not self.__eq__(other)

class Intersections:

	def __init__(self):
		self.intersections = []
		self.nearest_positive = -1

	def add(self, i):
		self.intersections.append(i)
		if i.t >= 0 and (self.nearest_positive < 0 or self.intersections[self.nearest_positive].t > i.t):
			self.nearest_positive = len(self.intersections) - 1

	def __len__(self):
		return len(self.intersections)

	def __getitem__(self, index):
		return self.intersections[index]

	def get_hit(self):
		if self.nearest_positive < 0:
			return None
		return self.intersections[self.nearest_positive]

class Sphere:

	def __init__(self):
		self.inverse_transform = xf_identity()
		self.material = default_material()

	def set_transform(self, m):
		self.inverse_transform = m.invert()

	def intersect(self, ray):
		ray = ray.transform(self.inverse_transform)
		sphere_to_ray = ray.origin.sub(point(0, 0, 0))
		a = ray.direction.dot(ray.direction)
		b = 2 * ray.direction.dot(sphere_to_ray)
		c = sphere_to_ray.dot(sphere_to_ray) - 1
		discriminant = b*b - 4*a*c
		if discriminant < 0:
			return Intersections()
		else:
			ii = Intersections()
			ii.add(Intersection(self, (-b - sqrt(discriminant)) / (2*a)))
			ii.add(Intersection(self, (-b + sqrt(discriminant)) / (2*a)))
			return ii



	def normal_at(self, p):
		object_point = p.transform(self.inverse_transform)
		object_normal = object_point.sub(point(0, 0, 0))
		world_normal = object_normal.transform(self.inverse_transform.transpose())
		world_normal.w = 0
		return world_normal.normalize()


def zero_array(size):
	a = []
	for i in range(0, size):
		a.append([0.0] * size)
	return a


def default_material():
	return Material(Color(1, 1, 1), 0.1, 0.9, 0.9, 200.0)

def lighting(material, light, point, eyev, normalv):
	effective_color = material.color.mul(light.color)
	lightv = light.position.sub(point).normalize()
	ambient = effective_color.mul_scalar(material.ambient)
	light_dot_normal = lightv.dot(normalv)
	if light_dot_normal < 0:
		diffuse = Color(0, 0, 0)
		specular = Color(0, 0, 0)
	else:
		diffuse = effective_color.mul_scalar(material.diffuse*light_dot_normal)

		reflectv = lightv.negate().reflect(normalv)
		reflect_dot_eye = reflectv.dot(eyev)
		if reflect_dot_eye < 0:
			specular = Color(0, 0, 0)
		else:
			factor = pow(reflect_dot_eye, material.shininess)
			specular = light.color.mul_scalar(material.specular * factor)
	return ambient.add(diffuse).add(specular)

class Material:

	def __init__(self, color, ambient, diffuse, specular, shininess):
		self.color = color
		self.ambient = ambient #background/base colors
		self.diffuse = diffuse #"lighted" color, from light source
		self.specular = specular #little light point/highlights
		self.shininess = shininess #shiny light shit

	

class Light:

		def __init__(self, color, position):
			self.color = color
			self.position = position


class Matrix:

	def __init__(self, initializer):
		if not isinstance(initializer, list):
			raise ValueError("bad initializer")
		if any(map(lambda l: len(l) != len(initializer), initializer)):
			raise ValueError("square matrices only")

		self.array = [map(lambda n: float(n), row) for row in initializer]

	def __eq__(self, other):
		if isinstance(other, Matrix):
			return self.array == other.array
		return false

	def __ne__(self, other):
		return not self.__eq__(other)

	def __repr__(self):
		return "\n".join(str(v) for v in self.array)

	def transpose(self):
		a = zero_array(len(self.array))
		for row in range(0, len(a)):
			for col in range(0, len(a)):
				a[row][col] = self.array[col][row]
		return Matrix(a)

	def mul(self, other):
		if not isinstance(other, Matrix):
			raise ValueError("")
		a = self.array
		b = other.array
		p = zero_array(len(a))
		for row in range(0, len(a)):
			for col in range(0, len(a)):
				sum = 0
				for i in range(0, len(a)):
					sum += a[row][i] * b[i][col]
				p[row][col] = sum
#				p[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col] + a[row][3] * b[3][col]

		return Matrix(p)

	def mul_tuple(self, t):
		if not isinstance(t, Tuple):
			raise ValueError("")
		if len(self.array) != 4:
			raise ValueError("incompatible sizes")
		a = self.array
		return Tuple(
			a[0][0]*t.x + a[0][1]*t.y + a[0][2]*t.z + a[0][3]*t.w,			
			a[1][0]*t.x + a[1][1]*t.y + a[1][2]*t.z + a[1][3]*t.w,
			a[2][0]*t.x + a[2][1]*t.y + a[2][2]*t.z + a[2][3]*t.w,
			a[3][0]*t.x + a[3][1]*t.y + a[3][2]*t.z + a[3][3]*t.w,
			)


	def determinant(self):
		a = self.array
		if len(a) == 2:
			return a[0][0] * a[1][1] - a[0][1] * a[1][0] 
		else:
			det = 0
			for col in range(0, len(a)):
				det += a[0][col] * self.cofactor(0, col)
			return det

	def submatrix(self, i, j):
		a = self.array
		if len(a) <= 2:
			raise ValueError("matrix is too small to take a submatrix")
		new_array = zero_array(len(a) - 1)
		for row in range(0, len(new_array)):
			for col in range(0, len(new_array)):
				new_array[row][col] = a[row if row < i else row+1][col if col < j else col+1] 
		return Matrix(new_array)

	def minor(self, i, j):
		return self.submatrix(i, j).determinant()

	def cofactor(self, i, j):
		m = self.minor(i, j)
		if (i+j) % 2 == 1:
			m = -1*m
		return m

	def is_invertable(self):
		return self.determinant() != 0

	def invert(self):
		if not self.is_invertable():
			raise ValueError("matrix is not invertable")
		new_array = zero_array(len(self.array))
		det = self.determinant()
		for row in range(0, len(new_array)):
			for col in range(0, len(new_array)):
				c = self.cofactor(row, col)
				new_array[col][row] = c / det
		return Matrix(new_array)

def xf_identity():
	return Matrix([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]
		])

def xf_translate(x, y, z):
	return Matrix([
		[1, 0, 0, x],
		[0, 1, 0, y],
		[0, 0, 1, z],
		[0, 0, 0, 1]
		])

def xf_scale(x, y, z):
	return Matrix([
		[x, 0, 0, 0],
		[0, y, 0, 0],
		[0, 0, z, 0],
		[0, 0, 0, 1]
		])

t1 = Tuple(3, -2, 5, 1)
t2 = Tuple(-2, 3, 1, 0)

t3 = t1.add(t2)



def first_render(): #use to get lame red sphere
	canvas = Canvas(100, 100)
	color = Color(1, 0, 0,)
	s = Sphere()
	wall_size = 7.0
	half = wall_size / 2
	pixel_size = wall_size / 100
	origin = point(0, 0, -5)

	for row in range(100):
		world_y = half - pixel_size*row
		for col in range(100):
			world_x = -half + pixel_size*col
			position = point(world_x, world_y, 10)
			ray = Ray(origin, position.sub(origin).normalize())
			xs = s.intersect(ray)
			if xs.get_hit():
				canvas.write_pixel(col, row, color)

	canvas.save("first.ppm")

if __name__ == '__main__':
	width = 200
	canvas = Canvas(width, width)
	s = Sphere()
	s.material.color = Color(0, 0, 1)#gonna make a purple look
	wall_size = 7.0
	half = wall_size / 2
	pixel_size = wall_size / width
	origin = point(0, 0, -5)

	light = Light(Color(1, 1, 1), point(-10, 10, -10))#code for the light source, coords place the light behind us and 1,1,1 makes it white

	for row in range(width):
		world_y = half - pixel_size*row
		for col in range(width):
			world_x = -half + pixel_size*col
			position = point(world_x, world_y, 10)
			ray = Ray(origin, position.sub(origin).normalize())
			xs = s.intersect(ray)
			hit = xs.get_hit() #at somepoint along the ray, there is a collision
			if hit:
				hit_point = ray.position(hit.t) #theres a ray, and somewhere along the line, a distance t, is where the hit is, now we need to find the point... #take a distance t to the point above, figures out coordinates where point is
				normalv = hit.shape.normal_at(hit_point) #which shape is hit? (only one for now) calcuate the point on the sphere that was hit and where the normal vector is, so to dtermine the angle of light reflection
 				eyev = ray.direction.negate() #want a vector that points back to the "eye" so simply a ngegative ray, hence the negate 
 				color = lighting(hit.shape.material, light, hit_point, eyev, normalv) #determines what the color, using phong model, see 240 for more
 
				canvas.write_pixel(col, row, color)

	canvas.save("second.ppm")




	