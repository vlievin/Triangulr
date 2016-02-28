import matplotlib.image as mpimg
import numpy as np
import random
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import numpy as np
import cv2

from scipy.spatial import Delaunay


def getUniformPoints( N , img ):
    n = img.shape[0]
    m = img.shape[1]
    mean = (1, 1)
    cov = [[1, 0], [0, 1]]
    points = np.random.multivariate_normal(mean, cov, N)
    for i in range(N):
        points[i , 0] =  random.choice(range( n ))
        points[i , 1] =  random.choice(range( m )) 
        
    points[0 , 0] = 0
    points[0 , 1] = 0
    points[1 , 0] = 0
    points[1 , 1] = m
    points[2 , 0] = n
    points[2 , 1] = 0
    points[3 , 0] = n
    points[3 , 1] = m
    
    pts_line = int(np.sqrt( N ))- 2
    k = 4
    for i in range(pts_line):
            points[k , 0] = random.choice(range( n ))
            points[k , 1] = 0
            k+=1
            points[k , 0] = 0
            points[k , 1] = random.choice(range( m )) 
            k+=1
            points[k , 0] = 1
            points[k , 1] = random.choice(range( m )) 
            k+=1
            points[k , 0] = random.choice(range( n ))
            points[k , 1] = 1
            k+=1
    
    return points

def getPoint( x , i , factor_size = 1 ):
        return [ factor_size * int( x[ i , 1]) ,  factor_size * int(x[ i , 0] )]
    
def getTriangulation( points, img, filename , factor_size = 1):
    result = np.zeros((factor_size * img.shape[0], factor_size * img.shape[1],3), np.uint8)
    tri = Delaunay(points)

    def getColor( tri , img ):
        x = float( points[ tri[0] , 0 ] +points[ tri[1] , 0 ] + points[ tri[2] , 0 ]   ) / float(3)
        y = float( points[ tri[0] , 1 ] +points[ tri[1] , 1 ] + points[ tri[2] , 1 ]   ) / float(3)
        v = img[ x , y ]
        return  ( float(v[0]) , float(v[1])  , float(v[2])  )

    for t in tri.simplices:
        n = img.shape[0]
        m = img.shape[1]
        col = getColor( t , img )
        polygon = np.vstack(( getPoint( points, t[0] , factor_size), getPoint( points, t[1] , factor_size)))
        polygon = np.vstack(( polygon, getPoint( points, t[2] , factor_size )  ))
        cv2.fillConvexPoly(result, polygon, col, cv2.CV_AA, 0);
        
    return points, result
        
def trianglulate( filename, img , N = 1000 , points = [] , display = True, factor_size = 1):
    
    n = img.shape[0]
    m = img.shape[1]
    kx = 1.0
    ky = 1.0
    if n > m:
        ky = float(n) / float(m)
    if m > n:
        kx = float(m) / float(n)
        
    if not len(points):
        points = getUniformPoints( N , img )
    
    return getTriangulation( points, img, filename, factor_size = factor_size )


def getRandPoint( pt , r ):
    MAX = 1000
    theta = 2* 3.14 * float(random.choice( range(MAX) )) / float(MAX)
    res =( int(pt[0] +  r * np.cos( theta )) , int(pt[1] + r * np.sin(theta) ))
    return res

def generateContourPoints( img, contour , r,  points = [] ):
    forbidden = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    for pt in points:
        cv2.circle(forbidden ,(int(pt[1]), int(pt[0]))  , r, (255,255,255), -1)
    for pt in contour:
        new_pt = getRandPoint(pt, r)
        while new_pt[0] < 0 or new_pt[1] < 0 or new_pt[0] >= img.shape[0] or new_pt[1] >= img.shape[1]:
            new_pt = getRandPoint(pt, r)
        f = forbidden[  new_pt[0] , new_pt[1] , : ]
        if f[0] == 0:
            points.append( new_pt)
            cv2.circle(forbidden , ( int(new_pt[1]), int(new_pt[0]) ) , r, (255,255,255), -1)
    return points

def getNiceTriangulation( filename, N = 100 , r = 8, size = 600, factor_size = 1 ):
    BLUR_SIZE = 4
    img=mpimg.imread(filename)
    #img = cv2.blur(img,(BLUR_SIZE,BLUR_SIZE),0)
    height, width = img.shape[:2]
    k = float(size) / float(height)
    if k > 1:
        k = 1
    img = cv2.resize(img,( int(k*width), int(k*height)), interpolation = cv2.INTER_CUBIC)
    edges = cv2.Canny(img ,100,300)
    contour = []
    for i in range( edges.shape[0]):
        for j in range( edges.shape[1]):
            if edges[i,j] > 0:
                contour.append((i,j))
    np_points = getUniformPoints( 100 , img )
    points = np_points.tolist()
    points = generateContourPoints( img, contour, r , points = points)
    points = np.array( points)
    return trianglulate( filename, img , N = 1000 , points = points ,factor_size = factor_size)


