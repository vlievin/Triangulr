import matplotlib.image as mpimg
import random
import scipy
import numpy as np

from scipy import ndimage as ndi
import skimage
from skimage import feature
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
    
def getTriangulation( points, img, factor_size = 1):
    result = np.zeros((factor_size * img.shape[0], factor_size * img.shape[1], 3 ), np.uint8)
    tri = Delaunay(points)
    
    def getColor( tri , img ):
        x = float( points[ tri[0] , 0 ] +points[ tri[1] , 0 ] + points[ tri[2] , 0 ]   ) / float(3)
        y = float( points[ tri[0] , 1 ] +points[ tri[1] , 1 ] + points[ tri[2] , 1 ]   ) / float(3)
        v = img[ x , y ]
        return  ( 255.0 * float(v[0]) ,255.0 * float(v[1])  ,255.0 * float(v[2])  )

    for t in tri.simplices:
        n = img.shape[0]
        m = img.shape[1]
        col = getColor( t , img )
        polygon = np.vstack(( getPoint( points, t[0] , factor_size), getPoint( points, t[1] , factor_size)))
        polygon = np.vstack(( polygon, getPoint( points, t[2] , factor_size )  ))
        #print polygon[:, 0]
        #print polygon[:, 1]
        rr, cc = skimage.draw.polygon( polygon[:, 1], polygon[:, 0] )
        #print col
        result[rr, cc, 0] = col[0]
        result[rr, cc, 1] = col[1]
        result[rr, cc, 2] = col[2] 
        
    return points, result



def getRandPoint( pt , r ):
    MAX = 1000
    theta = 2* 3.14 * float(random.choice( range(MAX) )) / float(MAX)
    res =( int(pt[0] +  r * np.cos( theta )) , int(pt[1] + r * np.sin(theta) ))
    return res

def generateContourPoints( img, contour , r,  points = [] ):
    forbidden = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    for pt in points:
        #cv2.circle(forbidden ,(int(pt[1]), int(pt[0]))  , r, (255,255,255), -1)
        rr, cc = skimage.draw.circle( int(pt[0]), int(pt[1]), r, forbidden.shape)
        forbidden[rr, cc] = 1
    for pt in contour:
        new_pt = getRandPoint(pt, r)
        while new_pt[0] < 0 or new_pt[1] < 0 or new_pt[0] >= img.shape[0] or new_pt[1] >= img.shape[1]:
            new_pt = getRandPoint(pt, r)
        f = forbidden[  new_pt[0] , new_pt[1] ]
        if f == 0:
            points.append( new_pt )
            rr, cc = skimage.draw.circle( int(new_pt[0]) , int(new_pt[1]) , r, forbidden.shape )
            forbidden[rr, cc] = 1
    #plt.imshow( forbidden, cmap = cm.Greys_r )
    #plt.axis('off')
    #plt.show()
    return points



def getContourPoints(img):
    gray = skimage.color.rgb2gray(img)
    edges = feature.canny(gray, sigma=1.7)
    contour = []
    for i in range( edges.shape[0]):
        for j in range( edges.shape[1]):
            if edges[i,j] > 0:
                contour.append((i,j))
    return contour

def resize(img, size):
    height, width = img.shape[:2]
    k = float(size) / float(height)
    if k > 1:
        k = 1
    return skimage.transform.resize(img,( int(k*height), int(k*width)))


def getNiceTriangulation( filename, N = 100 , r = 5, size = 600, factor_size = 1 ):
    img=mpimg.imread(filename)
    img = resize(img, size)
    contour = getContourPoints(img)
    np_points = getUniformPoints( 100 , img )
    points = np_points.tolist()
    pts = generateContourPoints(img, contour , r, points = points)
    points = np.array( points)
    return getTriangulation( points, img, factor_size = 1)