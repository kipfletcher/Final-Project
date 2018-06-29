#  Kip Fletcher  MATH 4330  6-26-2018
#  Final Project - Least Squares

#function that returns the product of a matrix and a vector.
def matVec(matrix, vector):
    '''
    This function takes a matrix and a vector as its arguments. It then
    creates a new matrix by multiplying the input matrix by the input vector
    and returns the new matrix. 
    '''
    new_matrix = []         # holds the matrix vector product.
    if len(matrix[0]) == len(vector):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if type(matrix[i][j]) != int and type(matrix[i][j]) != float and type(matrix[i][j]) != complex:
                   print("Invalid matrix element in matVec function.")
        for i in vector:
             if type(i) != int and type(i) != float and type(i) != complex:
                   print("Invalid vector element in matVec function.")
    
        for i in range(len(matrix)):
            row_product = 0     # holds the product of each matrix row.
            for j in range(len(vector)):
                row_product += (matrix[i][j]*vector[j])
            new_matrix.append(row_product) # adds each row product entry to new matrix.
        return new_matrix
    else:
        print("Incompatible matrix-vector in matVec function.")
        
#function that returns the difference of two vectors.
def vec_sub(vector01,vector02):
    '''
    This function take two vectors, verifies that they are the same length with
    an if-else statement, and subtracts them using a for loop. It returns the
    difference of the two vectors as a new vector.
    '''

    if len(vector01) == len(vector02):
        for i in vector01:
            if type(i) != int and type(i) != float and type(i) != complex:
               print("Invalid vector element in vec_sub function.")  
        for i in vector02:
            if type(i) != int and type(i) != float and type(i) != complex:
               print("Invalid vector element in vec_sub function.") 
        b = []
        j = len(vector01)
        for i in range(j):
         b.append(vector01[i]-vector02[i])
        return b
    else:
      return "Invalid vector lengths for vec_sub function"  

#function that returns the product of a vector and a scalar.
def scavec(vector,scalar):
    '''
    This function take a vector and a scalar as its arguments, multiplies them
    together with a for loop, and returns the product as a new vector.
    '''
    b = []
    for i in vector:
         b.append(i * scalar)

    return b

#function that returns the dot product of two vectors.
def dot(vector01, vector02):
    '''
    This function takes two compatible vectors as its arguments and returns
    their dot product. It first uses an if-else statement to verify that both
    vectors are vectors. It utilizes an if-else statement to determine if the two
    vectors are compatible and a for loop computes the dot product of the two
    vectors. If the two vectors are not compatible it returns "Invalid
    Input."
    '''
    dot_prod = 0

    if len(vector01) == len(vector02):
        for i in vector01:
            if type(i) != int and type(i) != float and type(i) != complex:
               print("Invalid vector element in dot function.")  
        for i in vector02:
            if type(i) != int and type(i) != float and type(i) != complex:
               print("Invalid vector element in dot function.") 
        for i in range(len(vector01)): 
            dot_prod += vector01[i]*vector02[i]
        return dot_prod
    else:
        print("Invalid Input")

#function that returns the 2-norm of a vector.
def norm_2(vector):
    '''
    This function takes a vector and computes its 2-norm using a for loop. It
    computes the sum of the squares of the absolute values of the vector elements.
    Then it takes the square root of that sum and returns its value as the 2-norm.
    '''
    sum1 = 0
    norm = 0
    for i in range(len(vector)):
        sum1 += (vector[i]) ** 2
    norm = sum1 ** (1/2)
    return norm

#function that returns the transpose of a matrix.
def matrix_transposer(matrix01):
    '''
    This function takes a single matrix and computes its transpose. It uses nested
    for loops to move each entry of the matrix to its corresponding position in
    the transposed matrix. It then returns the transpose of the original matrix.
    '''

    matrix01_trans = []

    for i in range(len(matrix01[0])):
        new_row = []
        for j in range(len(matrix01)):
            new_row.append(matrix01[j][i])
        matrix01_trans.append(new_row)

    return matrix01_trans

#function that returns the Reduced QR factorization of a matrix.
def gramSchmitt_mod(matrix):
    '''
    This function takes a matrix and decomposes it into a reduced orthogonal
    matrix q and a reduced upper triangular matrix r using the modified Gram
    Schmitt process. It first transposes the matrix to put the column vectors
    into rows and then creates three zero matrices of appropriate size to hold
    a copy of the input matrix, the q matrix, and the r matrix.  It uses a
    for-loop to copy the input matrix into matrix v, and then nested for-loops
    to produce matrices q and r.  To do this it implements 4 other functions:
    the norm_2, scavec,dot and vec_sub functions. It also contains an if-else
    statement to validate that none of the main diagonal entries of the r matrix
    are zero.
    '''
    
    rows = len(matrix)
    cols = len(matrix[0])
    a = []
    a = matrix_transposer(matrix)

    v = [[0]*rows]
    for i in range(cols-1):
        v.append([0]*cols)

    r = [[0]*cols]
    for i in range(cols-1):
        r.append([0]*cols)

    q = [[0]*rows]
    for i in range(cols-1):
        q.append([0]*cols)

    for i in range(0,cols,1):
        v[i] = a[i]
         
    for i in range(0,cols,1):
        r[i][i] = norm_2(v[i])
        if r[i][i] != 0:
          q[i] = scavec(v[i],(1/r[i][i]))
        else:
          print("Error: R values on main diagonal cannot be 0")
        for j in range(i+1,cols,1):
            r[i][j] = dot(q[i],v[j])
            v[j] = vec_sub(v[j],scavec(q[i],r[i][j]))
    
    qt = matrix_transposer(q)
    '''
    #print("Q = ")
    for i in range(len(q[0])):
      #print(qt[i])
    #print("R = ")
    for j in range(len(r)):
       #print(r[j])
    '''
   
    list1 = [qt,r]
    return list1

#function that returns the product of reduced matrix Q and the output vector 
def q_b(matrix,vector):
    '''
    This function takes a matrix and a vector as its arguments and validates
    them with for-loops and if-else statements. It uses the gramSchmitt_mod
    function to determine the Q matrix of the input matrix. It then uses the
    matrix_transposer function to transpose the Q matrix of the input matrix.
    Lastly,  it implements the matVec function to multiply the transposed Q
    matrix by the ouput vector b, of the equation Ax=b, and returns the product
    as a new vector.
    '''
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if type(matrix[i][j]) != int and type(matrix[i][j]) != float and type(matrix[i][j]) != complex:
               print("Invalid matrix element in q_b function.")
    for i in vector:
        if type(i) != int and type(i) != float and type(i) != complex:
           print("Invalid vector element in q_b function.")  
        
    new_vec = []
    q = gramSchmitt_mod(matrix)[0]
    q_trans = matrix_transposer(q)
    if len(q_trans[0]) == len(vector):
        new_vec = matVec(q_trans,vector)
        return new_vec
    else:
        print("Invalid matrix-vector combination in q_b function.")

#function that returns unknown c vector of the equation Ac=y.
def back_sub(matrix,vector):
    '''
    This function takes a matrix A and a vector y of the linear equation Ac=y
    and uses the back-substitution method to determine the unknown vector c. It
    uses if-else statements with for-loops to validate the input matrix and vector.
    It uses nested for-loops to calculate the elements of c and returns the c vector.
    '''
    if len(matrix) == len(vector):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if type(matrix[i][j]) != int and type(matrix[i][j]) != float and type(matrix[i][j]) != complex:
                   print("Invalid matrix element in back_sub function.")
        for i in vector:
             if type(i) != int and type(i) != float and type(i) != complex:
                   print("Invalid vector element in back_sub function.")
        c = vector
        m = len(matrix)
        for i in range(m-1,-1,-1):
           for j in range(i+1,m):
               c[i] = c[i] - (matrix[i][j]*c[j])
           c[i] = c[i]/matrix[i][i]
        return c
    else:
        print("Incompatible matrix-vector in back_sub function.")

#These are testcases for the matVec function

#These are testcases for the vec_sub function

#These are testcases for the scavec function

#These are testcases for the dot function

#These are testcases for the norm_2 function

#These are testcases for the matrix_transposer function

#These are testcases for the gramSchmitt_mod function

#These are testcases for the back_sub function

#These are testcases for the q_b function        

        
#This is the Vandermonde matrix A with input data.
A = [[1,.55,(.55)**2,(.55)**3],
      [1,.60,(.60)**2,(.60)**3],
      [1,.65,(.65)**2,(.65)**3],
      [1,.70,(.70)**2,(.70)**3],
      [1,.75,(.75)**2,(.75)**3], 
      [1,.80,(.80)**2,(.80)**3],
      [1,.85,(.85)**2,(.85)**3],
      [1,.90,(.90)**2,(.90)**3],
      [1,.95,(.95)**2,(.95)**3],
      [1,1.00,   1.00,   1.00,]]

#This is the output vector y of the equation Ac=y.
y = [1.102, 1.099, 1.017, 1.111, 1.117, 1.152, 1.265, 1.380, 1.575, 1.875]

#This vector is the product Q*y of the equation Rc=Q*y where A=QR.
p = []
p = q_b(A,y)

#This is the Q matrix of the equation A=QR.
q = []
q = gramSchmitt_mod(A)[0]

#This is the R matrix of the equation A=QR.
r = []
r = gramSchmitt_mod(A)[1]

#This is the c vector of the equation Ac=y.
c = []
c = back_sub(r,p) 
p = q_b(A,y) #reassigns the value q_b(m2,v) to y after the y value is
              #altered by the previous statement: x = back_sub(r,y)

print("A = ")
for i in range(len(A)):
    print(A[i])
print("Q = ")
for i in range(len(q)):
    print(gramSchmitt_mod(A)[0][i])
print("R = ")
for i in range(len(r)):
    print(r[i])
print()
print("c =",c)
print()
print("f(x)=",c[0],"+",c[1],"*x","+", c[2],"*x**2 +",c[3],"*x**3")



    



