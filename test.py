import numpy as np
#########################
#       Exercise 1      #
#########################

def generateTranslationMatrix(x, y, z):
    '''
    return the homogeneous transformation matrix for the given translation (x, y, z)
      parameter: 
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    # build a 3D Homogeneous Translation (Using a matrix Muliplication) 
    t_matrix = np.eye(4)
    t_matrix[0,3] = x
    t_matrix[1,3] = y
    t_matrix[2,3] = z

    return t_matrix


def generateScalingMatrix(sx, sy, sz):
    '''
    return the homogeneous transformation matrix for the given scaling parameters (sx, sy, sz)
      parameter:
        sx, sy, sz: scaling parameters for x-, y-, z-axis, respectively
      return:
        ndarray of size (4, 4)
    '''
    # build a 3D Homogeneous Scaling (Using a matrix Muliplication) 
    s_matrix = np.eye(4)
    s_matrix[0,0] = sx
    s_matrix[1,1] = sy
    s_matrix[2,2] = sz
    
    return s_matrix

def generateRotationMatrix(rad, axis):
    '''
    return the homogeneous transformation matrix for the given rotation parameters (rad, axis)
      parameter:
        rad: radians for rotation
        axis: axis for rotation, can only be one of ('x', 'y', 'z', 'X', 'Y', 'Z')
      return: 
        ndarray of size (4, 4)
    '''
    # 3D Homogeneous Rotation
    sin_rad = np.sin(rad)
    cos_rad = np.cos(rad)

    r_matrix = np.eye(4)
    if axis == 'x' or axis == 'X': # X-Axis Rotation Matrix
      r_matrix[1][1] = cos_rad
      r_matrix[1][2] = -sin_rad
      r_matrix[2][1] = sin_rad
      r_matrix[2][2] = cos_rad
      return r_matrix
    elif axis == 'y' or axis == 'Y': # Y-Axis Rotation Matrix
      r_matrix[0][0] = cos_rad
      r_matrix[0][2] = sin_rad
      r_matrix[2][0] = -sin_rad
      r_matrix[2][2] = cos_rad
      return r_matrix
    elif axis == 'z' or axis == 'Z': # Z-Axis Rotation Matrix
      r_matrix[0][0] = cos_rad
      r_matrix[0][1] = -sin_rad
      r_matrix[1][0] = sin_rad
      r_matrix[1][1] = cos_rad
      return r_matrix


# Case 1
def part1Case1():
    # translation matrix
    t = generateTranslationMatrix(2,3,-2)
    # scaling matrix
    s = generateScalingMatrix(0.5,2,2)
    # rotation matrix
    r = generateRotationMatrix(np.pi/4,'z')
    # data in homogeneous coordinate
    data = np.array([2, 3, 4, 1]).T
    after_t = np.dot(t,data)
    after_s = np.dot(s,after_t)
    after_r = np.dot(r,after_s)

    print("Case 1:\nafter t:")
    print(after_t)
    print("after s:")
    print(after_s)
    print("after r:")
    print(after_r)

# Case 2
def part1Case2():
    # translation matrix
    t = generateTranslationMatrix(4,-2,3)
    # scaling matrix
    s = generateScalingMatrix(3,1,3)
    # rotation matrix
    r = generateRotationMatrix(-np.pi/6,'y')
    # data in homogeneous coordinate
    data = np.array([6, 5, 2, 1]).T
    after_s = np.dot(s,data)
    after_t = np.dot(t,after_s)
    after_r = np.dot(r,after_t)

    print("Case 2:\nafter s:")
    print(after_s)
    print("after t:")
    print(after_t)
    print("after r:")
    print(after_r)

# Case 3
def part1Case3():
    # translation matrix
    t = generateTranslationMatrix(5,2,-3)
    # scaling matrix
    s = generateScalingMatrix(2,2,-2)
    # rotation matrix
    r = generateRotationMatrix(np.pi/12,'x')
    # data in homogeneous coordinate
    data = np.array([3, 2, 5, 1]).T
    after_r = np.dot(r,data)
    after_s = np.dot(s,after_r)
    after_t = np.dot(t,after_s)

    print("Case 3:\nafter r:")
    print(after_r)
    print("after s:")
    print(after_s)
    print("after t:")
    print(after_t)


#########################
#       Exercise 2      #
#########################

# Part 1
def generateRandomSphere(r, n):
    '''
    generate a point cloud of n points in spherical coordinates (radial distance, polar angle, azimuthal angle)
      parameter:
        r: radius of the sphere
        n: total number of points
    return:
      spherical coordinates, ndarray of size (3, n), 
      where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    '''
    # This was adapted from a post of Wikipedia named "Spherical coordinate system" 
    # forum here: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    # compute the spherical coordinates with radial distance, polar angle and azimuthal angle
    sphere = np.zeros((3,int(n)))
    radial_diatance = r
    polar_angle = np.pi
    azimuthal_angle = 2 * np.pi

    sphere[0,:] = np.random.rand(n) * radial_diatance  # r
    sphere[1,:] = np.random.rand(n) * polar_angle      # theta
    sphere[2,:] = np.random.rand(n) * azimuthal_angle  # phi
    
    return sphere

def sphericalToCatesian(coors):
    '''
    convert n points in spherical coordinates to cartesian coordinates, then add a row of 1s to them to convert
    them to homogeneous coordinates
      parameter:
        coors: ndarray of size (3, n), where the 3 rows are ordered as (radial distances, polar angles, azimuthal angles)
    return:
      catesian coordinates, ndarray of size (4, n), where the 4 rows are ordered as (x, y, z, 1)
    '''
    # This was adapted from a post of Wikipedia named "Spherical coordinate system" 
    # forum here: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    # compute the catesian coordinatesd with r, theta, phi
    n = coors.shape[1] 
    added_row = np.ones(n) # compute a row of 1 
    x = np.zeros(n) 
    y = np.zeros(n)
    z = np.zeros(n)

    for i in range(n):
      x[i] = coors[0,i] * np.cos(coors[2,i]) * np.sin(coors[1,i])
      y[i] = coors[0,i] * np.sin(coors[2,i]) * np.sin(coors[1,i])
      z[i] = coors[0,i] * np.cos(coors[1,i])

    new_matrix = np.concatenate((x,y,z,added_row)) # compute a 1D array with required elements
    new_matrix = new_matrix.reshape(4,n) # transfer the 1D array to 2D

    return new_matrix


# Part 2
def applyRandomTransformation(sphere1):
    '''
    generate two random transformations, one of each (scaling, rotation),
    apply them to the input sphere in random order, then apply a random translation,
    then return the transformed coordinates of the sphere, the composite transformation matrix,
    and the three random transformation matrices you generated
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
      return:
        a tuple (p, m, t, s, r)
        p: transformed homogeneous coordinates, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        m: composite transformation matrix, ndarray of size (4, 4)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    '''
    x = np.random.rand(1)
    y = np.random.rand(1)
    z = np.random.rand(1)

    translate_factor = np.random.randint(-10,10)
    scaling_factor = np.random.choice([-9,-8,-7,-6,-5,-4,-3,-2,2,3,4,5,6,7,8,9])
    rotating_num = np.random.choice([-9,-8,-7,-6,-5,-4,-3,3,4,5,6,7,8,9])
    rotating_axis = np.random.choice(['x','y','z'])

    t = generateTranslationMatrix(translate_factor * x,translate_factor * y,translate_factor * z)
    s = generateScalingMatrix(scaling_factor * x, scaling_factor * y,scaling_factor * z)
    r = generateRotationMatrix(np.pi/rotating_num, rotating_axis)
    
    num = np.random.choice([1,2])
    if num == 1:  # scaling first followed by rotating and transformation
      t_r_s = np.dot(t,np.dot(r,s))
      m = t_r_s
    elif num == 2: # rotating first followed by scaling and transformation
      t_s_r = np.dot(t,np.dot(s,r))
      m = t_s_r
    p = np.dot(m,sphere1)

    for i in range(m.shape[0]):    # if the number in the matrix is very small and close to 0
      for j in range(m.shape[1]):  #  sign the number to be 0
        if np.isclose(m[i][j], 0):
          m[i][j] = 0

    return (p,m,t,s,r)


def calculateTransformation(sphere1, sphere2):
    '''
    calculate the composite transformation matrix from sphere1 to sphere2
      parameter:
        sphere1: homogeneous coordinates of sphere1, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
        sphere2: homogeneous coordinates of sphere2, ndarray of size (4, n), 
                 where the 4 rows are ordered as (x, y, z, 1)
    return:
      composite transformation matrix, ndarray of size (4, 4) 
    '''
    # u is 4*4, sigma is 4*n, v is n*n
    # This was adapted from a post named "Deep Learning Book Series · 2.9 The Moore Penrose Pseudoinverse" 
    # which is post on 26-03-2018 by Hadrienj
    # forum here: https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/
    
    # since M = U*sigma*VT, M+ = V*(sigma+)*UT, sigma*(sigma+) = I
    # therefore, after we multiply M and M+ together, we will get M*(M+) = I(the identity matrix)
    # since M' = HM, we can multiply M+ to the both side
    # which gives: M'M+ = HMM+ = H (the M*(M+) = I)
    # it shows that M'M+ = H
    u,sigma,vh = np.linalg.svd(sphere1)
    sigma_plus = np.zeros((sphere1.shape[0],sphere1.shape[1])).T
    sigma_plus[:sigma.shape[0],:sigma.shape[0]] = np.linalg.inv(np.diag(sigma))
    M_plus = np.dot(vh.T,np.dot(sigma_plus,u.T)) 
    H_matrix = np.dot(sphere2,M_plus)

    for i in range(H_matrix.shape[0]):   #  if the number in the matrix is very small and close to 0
      for j in range(H_matrix.shape[1]): #  sign the number to be 0
        if np.isclose(H_matrix[i][j], 0):
          H_matrix[i][j] = 0

    return H_matrix

def decomposeTransformation(m):
    """
    decomposite the transformation and return the translation, scaling, and rotation matrices
      parameter:
        m: homogeneous transformation matrix, ndarray of size (4, 4)

    return:
      tuple of three matrices, (t, s, r)
        t: translation matrix, ndarray of size (4, 4)
        s: scaling matrix, ndarray of size (4, 4)
        r: rotation matrix, ndarray of size (4, 4)
    """
    # This was adapted from a post on a website called "MATHEMATICS"
    # forum here: https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
        if np.isclose(m[i][j], 0):
          m[i][j] = 0
  
    n = m.shape[1] # n is 4
    t = np.eye(4)
    s = np.eye(4)
    r = np.eye(4)

    t[0,n-1] = m[0,n-1]
    t[1,n-1] = m[1,n-1]
    t[2,n-1] = m[2,n-1]
    
    double_sx = m[0,0]**2 + m[1,0]**2 + m[2,0]**2
    double_sy = m[0,1]**2 + m[1,1]**2 + m[2,1]**2
    double_sz = m[0,2]**2 + m[1,2]**2 + m[2,2]**2
    double_list = [double_sx,double_sy,double_sz]
    sqrt_list = np.sqrt(double_list)    # compute the norm of each column
    if (np.sign(m[0,0])!=0):    # give s matrix the correct sign, avoiding value of m[0,0] is 0
      s[0,0] = sqrt_list[0] * np.sign(m[0,0])
      s[1,1] = sqrt_list[1] * np.sign(m[0,0])
      s[2,2] = sqrt_list[2] * np.sign(m[0,0])
    elif (np.sign(m[1,1])!=0): # give s matrix the correct sign, avoiding value of m[1,1] is 0
      s[0,0] = sqrt_list[0] * np.sign(m[1,1])
      s[1,1] = sqrt_list[1] * np.sign(m[1,1])
      s[2,2] = sqrt_list[2] * np.sign(m[1,1])
    elif (np.sign(m[2,2])!=0): # give s matrix the correct sign, avoiding value of m[2,2] is 0
      s[0,0] = sqrt_list[0] * np.sign(m[2,2])
      s[1,1] = sqrt_list[1] * np.sign(m[2,2])
      s[2,2] = sqrt_list[2] * np.sign(m[2,2])

    if (m[0,1]!=0 and m[1,0]!=0): # rotate on z
      if (np.isclose([np.abs(m[0,0]/m[1,1])],[np.abs(m[0,1]/m[1,0])])): # t_s_r
        alpha = np.arctan(m[1,0]/m[1,1])
        r[0, 0] = np.cos(alpha)
        r[0, 1] = -np.sin(alpha)
        r[1, 0] = np.sin(alpha)
        r[1, 1] = np.cos(alpha)
        s[0,0] = m[0,0]/r[0,0]
        s[1,1] = m[1,1]/r[1,1]
        s[2,2] = m[2,2]
      elif (np.isclose([np.abs(m[0,0]/m[1,1])],[np.abs(m[1,0]/m[0,1])])): # t_r_s
        r[0,0] = m[0,0]/s[0,0]
        r[0,1] = m[0,1]/s[1,1]
        r[1,0] = m[1,0]/s[0,0]
        r[1,1] = m[1,1]/s[1,1]
    elif (m[1,2]!=0 and m[2,1]!=0): # rotate on x
      if (np.isclose([np.abs(m[1,1]/m[2,2])],[np.abs(m[1,2]/m[2,1])])): # t_s_r
        alpha = np.arctan(m[2,1]/m[2,2])
        r[1, 1] = np.cos(alpha)
        r[1, 2] = -np.sin(alpha)
        r[2, 1] = np.sin(alpha)
        r[2, 2] = np.cos(alpha)
        s[0, 0] = m[0, 0] 
        s[1, 1] = m[1, 1]/r[1, 1]
        s[2,2] = m[2,2]/r[2,2]
      elif (np.isclose([np.abs(m[1,1]/m[2,2])],[np.abs(m[2,1]/m[1,2])])): # t_r_s
        r[1,1] = m[1,1]/s[1,1]
        r[2,1] = m[2,1]/s[1,1]
        r[1,2] = m[1,2]/s[2,2]
        r[2,2] = m[2,2]/s[2,2]
    elif (m[0,2]!=0 and m[2,0]!=0): # rotate on y
      if (np.isclose([np.abs(m[0,0]/m[2,2])],[np.abs(m[0,2]/m[2,0])])): # t_s_r
        alpha = np.arctan(m[0,2]/m[0,0])
        r[0, 0] = np.cos(alpha)
        r[0, 2] = np.sin(alpha)
        r[2, 0] = -np.sin(alpha)
        r[2, 2] = np.cos(alpha)
        s[0, 0] = m[0, 0]/r[0,0] 
        s[1, 1] = m[1, 1]
        s[2,2] = m[2,2]/r[2,2]
      elif (np.isclose([np.abs(m[0,0]/m[2,2])],[np.abs(m[2,0]/m[0,2])])): # t_r_s
        r[0,0] = m[0,0]/s[0,0]
        r[0,2] = m[0,2]/s[2,2]
        r[2,0] = m[2,0]/s[0,0]
        r[2,2] = m[2,2]/s[2,2]

    return (t,s,r)

#########################
#      Main function    #
#########################
def main():
    print("Exercise 1")
    part1Case1()
    part1Case2()
    part1Case3()

    print("\nExercise 2")
    rotating_num = np.random.choice([3,4,5,6,7,8,9])
    random_n = np.random.randint(10,20) # the random n should always >= 4 to makes the two transformation matrix be the same
    random_sphere = generateRandomSphere(np.pi/rotating_num,random_n)
    coors = sphericalToCatesian(random_sphere)
    p,m,t,s,r = applyRandomTransformation(coors)
    trans_matrix_calculate  = calculateTransformation(coors,p)
    print("the first transformed matrix is :\n",m)
    print("the second transformed matrix is :\n",trans_matrix_calculate)
    if np.allclose(m,trans_matrix_calculate):
      print("- - - - - The two transformation matrix are the same! - - - - - ")
    else:
      print("- - - - - The two transformation matrix are the different! - - - - - ")
    
    decompose_t,decompose_s,decompose_r = decomposeTransformation(trans_matrix_calculate)
    print("the translation matrix is :\n",t)
    print("the decomposed translation matrix is : \n",decompose_t)
    print("the scaling matrix is :\n",s)
    print("the decomposed scaling matrix is : \n",decompose_s)
    print("the rotating matrix is :\n",r)
    print("the decomposed rotating matrix is : \n",decompose_r)
    if np.allclose(t, decompose_t) and np.allclose(s, decompose_s) and np.allclose(r, decompose_r):
      print("- - - - - The three components from decompositionare and that in applyRandomTransformation are the same! - - - - - ")
    else:
      print("- - - - - The three components from decompositionare and that in applyRandomTransformation are the different! - - - - - ")
    
if __name__ == "__main__":
    main()