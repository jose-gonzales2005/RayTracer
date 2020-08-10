import unittest 
from vector import Tuple, point, vector, Color, Canvas, Ray, Intersection, Intersections, Matrix, vector, point, xf_translate, xf_scale, Sphere
from math import sqrt

class MyTest(unittest.TestCase):

	def test_vec_add(self):

		t1 = Tuple(5, 3, -9, 0)
		t2 = Tuple(2, 7, -8, 3)
		self.assertEqual(t1.add(t2), Tuple(7, 10, -17, 3))

	def test_vec_sub(self):
		t1 = Tuple(1, -1, 0, 2)
		t2 = Tuple(3, 4, -6, -1)
		self.assertEqual(t1.sub(t2), Tuple(-2, -5, 6, 3))

	def test_vec_negate(self):
		t1 = Tuple(-9, 2, -1, 4)
		self.assertEqual(t1.negate(), Tuple(9, -2, 1, -4))

	def test_vec_mul(self):
		t1 = Tuple(1, 4, 5, -6)
		self.assertEqual(t1.mul(7), Tuple(7, 28, 35, -42))

	def test_vec_div(self):
		t1 = Tuple(9, 3, 12, 15)
		self.assertEqual(t1.div(2), Tuple(4.5, (1.5), 6, 7.5,))

	def test_magnitude1(self):
		v = vector(1, 0, 0)
		self.assertEqual(v.magnitude(), 1)

	def test_magnitude2(self):
		v = vector(1, 2, 3)
		self.assertEqual(v.magnitude(), sqrt(14))

	def test_normalize(self):
		v = vector(4, 0, 0)
		self.assertEqual(v.normalize(),vector(1, 0, 0))

	def test_dot1(self):
		self.assertEqual(vector(1, 2, 3).dot(vector(2, 3, 4)), 20)

	def test_cross1(self):
		v1 = vector(1, 2, 3)
		v2 = vector(2, 3, 4)
		self.assertEqual(v1.cross(v2), vector(-1, 2, -1))
		self.assertEqual(v2.cross(v1), vector(1, -2, 1))

	def test_canvas(self):
		c = Canvas(5, 3)
		c.write_pixel(0, 0, Color(1.5, 0, 0))
		c.write_pixel(2, 1, Color(0, 0.5, 0))
		c.write_pixel(4, 2, Color(-0.5, 0, 1))
		c.save("test1.ppm")

	def test_matrix_mul(self):
		m1 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6], [5, 4, 3, 2]])
		m2 = Matrix([[-2, 1, 2, 3], [3, 2, 1, -1], [4, 3, 6, 5], [1, 2, 7, 8]])
		answer = Matrix([[20, 22, 50, 48], [44, 54, 114, 108], [40, 58, 110, 102], [16, 26, 46, 42]])
		self.assertEqual(answer, m1.mul(m2))

	def test_matrix_mul_tuple(self):
		m = Matrix([
			[1, 2, 3, 4],
			[2, 4, 4, 2],
			[8, 6, 4, 1],
			[0, 0, 0, 1]
			])
		t = Tuple(1, 2, 3, 1);
		self.assertEqual(m.mul_tuple(t), Tuple(18, 24, 33, 1))

	def test_matrix_transpose(self):
		m1 = Matrix([[0, 9, 3, 0], [9, 8, 0, 8], [1, 8, 5, 3], [0, 0, 5, 8]])
		answer = Matrix([[0, 9, 1, 0], [9, 8, 8, 0], [3, 0, 5, 5], [0, 8, 3, 8]])
		self.assertEqual(answer, m1.transpose())

	def test_submatrix1(self):
		m1 = Matrix([[1, 5, 0], [-3, 2, 7], [0, 6, -3]])
		answer = Matrix([[-3, 2], [0, 6]])
		self.assertEqual(answer, m1.submatrix(0, 2))

	def test_submatrix2(self):
		m1 = Matrix([[-6, 1, 1, 6], [-8, 5, 8, 6], [-1, 0, 8, 2], [-7, 1, -1, 1]]);
		answer = Matrix([[-6, 1, 6], [-8, 8, 6], [-7, -1, 1]])
		self.assertEqual(answer, m1.submatrix(2, 1))

	def test_minor1(self):
		m1 = Matrix([[3, 5, 0], [2, -1, -7], [6, -1, 5]])
		self.assertEqual(m1.minor(1, 0), 25)

	def test_cofactor(self):
		m = Matrix([[3, 5, 0], [2,-1, -7], [6, -1, 5]])
		self.assertEqual(-12, m.minor(0, 0))
		self.assertEqual(-12, m.cofactor(0, 0))
		self.assertEqual(25, m.minor(1, 0))
		self.assertEqual(-25, m.cofactor(1, 0))

	def test_determinant1(self):
		m = Matrix([[1, 5], [-3, 2]])
		self.assertEqual(17, m.determinant())

	def test_determinant2(self):
		m = Matrix([[1, 2, 6], [-5, 8, -4], [2, 6, 4]])
		self.assertEqual(56, m.cofactor(0, 0))
		self.assertEqual(12, m.cofactor(0, 1))
		self.assertEqual(-46, m.cofactor(0, 2))
		self.assertEqual(-196, m.determinant())

	def test_determinant3(self):
		m = Matrix([[-2, -8, 3, 5], [-3, 1, 7, 3], [1, 2, -9, 6], [-6, 7, 7, -9]])
		self.assertEqual(690, m.cofactor(0, 0))
		self.assertEqual(447, m.cofactor(0, 1))
		self.assertEqual(210, m.cofactor(0, 2))
		self.assertEqual(51, m.cofactor(0, 3))
		self.assertEqual(-4071, m.determinant())


	def test_invertable1(self):
		m = Matrix([[6, 4, 4, 4], [5, 5, 7, 6], [4, -9, 3, -7], [9, 1, 7, -6]])
		self.assertEqual(-2120, m.determinant())
		self.assertTrue(m.is_invertable())

	def test_invertable2(self):
		m = Matrix([[-4, 2, -2, -3], [9, 6, 2, 6], [0, -5, 1, -5], [0, 0, 0, 0]])
		self.assertEqual(0, m.determinant())
		self.assertFalse(m.is_invertable())

	def test_inverse1(self):
		m = Matrix([[-5, 2, 6, -8], [1, -5, 1, 8], [7, 7, -6, -7], [1, -3, 7, 4]])
		self.assertEqual(532, m.determinant())
		self.assertEqual(160, m.minor(2, 3))
		self.assertEqual(-160, m.cofactor(2, 3))
		m_inverse = m.invert()

		answer = Matrix([
			[0.21805, 0.45113, 0.24060, -0.04511], 
			[-0.80827, -1.45677, -0.44361, 0.52068],
			[-0.07895, -0.22368, -0.05263, 0.19737],
			[-0.52256, -0.81391, -0.30075, 0.30639]])


	def test_inverse2(self):
		m = Matrix([[8, -5, 9, 2], [7, 5, 6, 1], [-6, 0, 9, 6], [-3, 0, -9, -4]])
		m_inverse = m.invert()


	def test_inverse3(self):
		m = Matrix([[9, 3, 0, 9], [-5, -2, -6, -3], [-4, 9, 6, 4], [-7, 6, 6, 2]])
		m_inverse = m.invert()

  	def test_translate1(self):
		p = point(-3, 4, 5)
		m = xf_translate(5, -3, 2)	
		self.assertEqual(p.transform(m), point(2, 1, 7))

  	def test_translate2(self):
		v = vector(-3, 4, 5)
		m = xf_translate(5, -3, 2)
		self.assertEqual(v.transform(m), vector(-3, 4, 5))

  	def test_translate3(self):
		p = point(-3, 4, 5)
		m = xf_translate(5, -3, 2)
		self.assertEqual(p.transform(m.invert()), point(-8, 7, 3))

	def test_scaling1(self):
		p = point(-4, 6, 8)
		m = xf_scale(2, 3, 4)
		self.assertEqual(p.transform(m), point(-8, 18, 32))

	def test_scaling2(self):
		v = vector(-4, 6, 8)
		m = xf_scale(2, 3, 4)
		self.assertEqual(v.transform(m), vector(-8, 18, 32))

	def test_scaling3(self):
		v = vector(-4, 6, 8)
		m = xf_scale(2, 3, 4).invert()
		self.assertEqual(v.transform(m), vector(-2, 2, 2))

	def test_ray_postion1(self):
		r = Ray(point(2, 3, 4), vector(1, 0, 0))
		self.assertEqual(r.position(0), point(2, 3, 4))
		self.assertEqual(r.position(1), point(3, 3, 4))
		self.assertEqual(r.position(-1), point(1, 3, 4))
		self.assertEqual(r.position(2.5), point(4.5, 3, 4))

	def test_intersect1(self):
		r = Ray(point(0, 0, -5), vector(0, 0, 1))
		s = Sphere()
		i = s.intersect(r)
		self.assertEqual(2, len(i))
		self.assertEqual(4.0, i[0].t)
		self.assertEqual(6.0, i[1].t)

	def test_intersect2(self):
		r = Ray(point(0, 1, -5), vector(0, 0, 1))
		s = Sphere()
		i = s.intersect(r)
		self.assertEqual(2, len(i))
		self.assertEqual(5.0, i[0].t)
		self.assertEqual(5.0, i[1].t)

	def test_intersect3(self):
		r = Ray(point(0, 2, -5), vector(0, 0, 1))
		s = Sphere()
		i = s.intersect(r)
		self.assertEqual(0, len(i))

	def test_intersect4(self):
		r = Ray(point(0, 0, 0), vector(0, 0, 1))
		s = Sphere()
		i = s.intersect(r)
		self.assertEqual(2, len(i))
		self.assertEqual(-1.0, i[0].t)
		self.assertEqual(1.0, i[1].t)

	def test_intersect5(self):
		r = Ray(point(0, 0, 5), vector(0, 0, 1))
		s = Sphere()
		i = s.intersect(r)
		self.assertEqual(2, len(i))
		self.assertEqual(-6.0, i[0].t)
		self.assertEqual(-4.0, i[1].t)

	def test_get_hit1(self):
		s = Sphere()
		ii = Intersections()
		i1 = Intersection(s, 1)
		ii.add(i1)
		i2 = Intersection(s, 2)
		ii.add(i2)
		self.assertEqual(ii.get_hit(), i1)

	def test_get_hit2(self):
		s = Sphere()
		ii = Intersections()
		i1 = Intersection(s, -1)
		ii.add(i1)
		i2 = Intersection(s, 1)
		ii.add(i2)
		self.assertEqual(ii.get_hit(), i2)

	def test_get_hit3(self):
		s = Sphere()
		ii = Intersections()
		i1 = Intersection(s, -2)
		ii.add(i1)
		i2 = Intersection(s, -1)
		ii.add(i2)
		self.assertEqual(ii.get_hit(), None)

	def test_get_hit4(self):
		s = Sphere()
		ii = Intersections()
		i1 = Intersection(s, 5)
		ii.add(i1)
		i2 = Intersection(s, 7)
		ii.add(i2)
		i3 = Intersection(s, -3)
		ii.add(i3)
		i4 = Intersection(s, 2)
		ii.add(i4)
		self.assertEqual(ii.get_hit(), i4)

	def test_translate_ray1(self):
		r = Ray(point(1, 2, 3), vector(0, 1, 0))
		m = xf_translate(3, 4, 5)
		r2 = r.transform(m)
		self.assertEqual(r2.origin, point(4, 6, 8))
		self.assertEqual(r2.direction, vector(0, 1, 0))

	def test_translate_ray2(self):
		r = Ray(point(1, 2, 3), vector(0, 1, 0))
		m = xf_scale(2, 3, 4)
		r2 = r.transform(m)
		self.assertEqual(r2.origin, point(2, 6, 12))
		self.assertEqual(r2.direction, vector(0, 3, 0))

	def test_intersect_sphere1(self):
		s = Sphere()
		r = Ray(point(0, 0, -5), vector(0, 0, 1))
		s.set_transform(xf_scale(2, 2, 2))
		i = s.intersect(r)
		self.assertEqual(2, len(i))
		self.assertEqual(i[0].t, 3.0)
		self.assertEqual(i[1].t, 7.0)

	def test_world_normal1(self):
		s = Sphere()
		s.set_transform(xf_translate(0, 1, 0))
		n = s.normal_at(point(0, 1.70711, -0.70711))
		#self.assertEqual(n, vector(0, 0.70711, -0.70711))

if __name__ == '__main__':
	unittest.main()