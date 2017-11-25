from Tkinter import Canvas, Tk, Frame, Button, RAISED, TOP, StringVar, Label, RIGHT, RIDGE
import random
import time
import math
import sys
from UnionFind import UnionFind
import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from descartes import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

tk = Tk()
tk.wm_title("Steiner Trees")

global pt
pt = [[0 for i in xrange(2)] for i in xrange(1000)]
global pt2
pt2 = []
global time1
time1 = []
global o_count
o_count = 0
global pt3
pt3 = []
global OriginalPoints
OriginalPoints = []
global RectSteinerPoints
RectSteinerPoints = []
global GraphSteinerPoints
GraphSteinerPoints = []
global Obstacle_pt
Obstacle_pt = []
global RSMT
RSMT = []
global GSMT
GSMT = []
global count
count = 0
global flag
flag = 0
global count23
count23 = 0
global max_xcoord
max_xcoord = []
global min_xcoord
min_xcoord = []
global max_ycoord
max_ycoord = []
global min_ycoord
min_ycoord = []

class Point:
    """Point Class for Steiner.py
    Contains position in x and y values with degree of edges representative of the length of
    the list of edges relative to the MST
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.deg = 0
        self.edges = []
        self.MSTedges = []
    def update(self, edge):
        self.edges.append(edge)
    def reset(self):
        self.edges = []
        self.deg = 0
        self.MSTedges = []
    def MSTupdate(self, edge):
        self.deg += 1
        self.MSTedges.append(edge)

class Line:
    """Line Class for Steiner.py
    Contains the two end points as well as the weight of the line. 
    Supports determining the first or last point as well as the other given one. 
    """
    def __init__(self, p1, p2, w):
        self.points = []
        self.points.append(ref(p1))
        self.points.append(ref(p2))
        self.w = w
    def getOther(self, pt):
        if pt == self.points[0].get():
            return self.points[1]
        elif pt == self.points[1].get():
            return self.points[0]
        else:
            print "This is an Error. The line does not contain points that make sense."
    def getFirst(self):
        return self.points[0]
    def getLast(self):
        return self.points[1]

class ref:
    """ref Class for use in Steiner.py
    Satisfies the need for pointers to maintain a constant and updated global list of things. 
    """
    def __init__(self, obj): 
        self.obj = obj
    def get(self):  
        return self.obj
    def set(self, obj):   
        self.obj = obj

def addMousePoint(event):
    """addMousePoint 
    Calls addPoint if point is not on canvas edge and not on top of another point. 
    """
    addpt = True
    if OriginalPoints == []:
        if (event.x < 10) and (event.x >= 800) and (event.y < 10) and (event.y >= 800):
                addpt = False
    else:
        for pt in OriginalPoints:
            dist = math.sqrt(pow((event.x - pt.x),2) + pow((event.y - pt.y),2))
            if dist < 11:
                addpt = False
            if (event.x < 10) and (event.x >= 800) and (event.y < 10) and (event.y >= 800):
                addpt = False
    if addpt ==  True:
            addPoint(event.x, event.y)
            
def addPoint(x, y):
    """addPoint 
    Adds a point at the specified x and y on the Tkinter canvas.
    """
    global flag
    global count
    global GSMT
    del GSMT[:]
    print x
    print y
    canvas.create_oval(x-5,y-5,x+5,y+5, outline="black", fill="white", width=1)
    if flag == 0:
        pt[count][0]=x
        pt[count][1]=y
        #Obstacle_pt.append(pt)
        count += 1
    else:
        point = Point(x, y)
        global OriginalPoints
        OriginalPoints.append(point)
    
def alpha_shape(pt, alpha):
    global count
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
               # already added
            return
        edges.add( (i, j) )
        edge_points.append(pt1[ [i, j] ])    
    #pt = np.array(pt)
    
    pto = np.array(pt)
    #print count
    #print pto
    pt1 = pto[0:count:1]
    #print pt1
    tri = Delaunay(pt1)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = pt1[ia]
        pb = pt1[ib]
        pc = pt1[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, pt1, ia, ib)
            add_edge(edges, edge_points, pt1, ib, ic)
            add_edge(edges, edge_points, pt1, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def alpha_sh() :
    global pt2
    #print pt
    """if o_count == 0:
	pt = [[363,335],[326,340],[302,369],[349,381],[380,357]]
    if o_count == 1:
	pt = [[529,304],[486,294],[460,322],[506,349],[535,332]]
    """
    
    concave_hull, edge_points = alpha_shape(pt, alpha=0.04) 
    print concave_hull  
    pt2 = np.array(concave_hull.exterior)
    length = len(pt2)
    for i in range(length-1):
        line=canvas.create_line(pt2[i][0],pt2[i][1],pt2[i+1][0],pt2[i+1][1])
    return pt2
    
def computeObstacle():
    t1 = time.time()
    global o_count
    global pt
    global count
    global pt3
    pt3.append(alpha_sh())
    o_count += 1
    count = 0
    pt = [[0 for i in xrange(2)] for i in xrange(1000)]  
    t2 = time.time()
    t3 = t2-t1
    time1.append(t3)
    
def completeObstacles():
    global flag
    flag = 1
    Obstext.set(o_count)
    for i in time1:
	print i
    t = max(time1)
    ObsTime.set(str(round(t, 6)))
    #ObsTime.set(str(round((t,6))))
    
def orientation1(j,i,r):
    global pt3
    val = (pt3[j][i+1][1] - pt3[j][i][1]) * (r.x - pt3[j][i+1][0]) - (pt3[j][i+1][0] - pt3[j][i][0]) * (r.y - pt3[j][i+1][1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2
    
def orientation2(p,q,j,i):
    global pt3
    val = (q.y - p.y) * (pt3[j][i][0] - q.x) - (q.x - p.x) * (pt3[j][i][1] - q.y)
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2    
    
    
def lineinpoly(pn1,pn2):
    for j in xrange(o_count):
        for i in xrange(len(pt3[j])-1):
            o1 = orientation1(j,i,pn1)
            o2 = orientation1(j,i,pn2)
            o3 = orientation2(pn1,pn2,j,i)
            o4 = orientation2(pn1,pn2,j,i+1)
            if (o1!=o2 and o3!=o4):
		return 0
    return 1

def lineinpolyRect(pn1,pn2):
    global min_xcoord
    global max_xcoord
    global min_ycoord
    global max_ycoord
    for j in xrange(o_count):
	if pn1.x <= max_xcoord[j] and pn1.x >= min_xcoord[j] and pn2.x <= max_xcoord[j] and pn2.x >= min_xcoord[j]:
	    for i in xrange(len(pt3[j])-1):
		if ((pn1.y < pt3[j][i][1] and pn2.y > pt3[j][i][1]) or (pn2.y < pt3[j][i][1] and pn1.y > pt3[j][i][1])):
		    return 0
	elif pn1.y <= max_ycoord[j] and pn1.y >= min_ycoord[j] and pn2.y <= max_ycoord[j] and pn2.y >= min_ycoord[j]:
	    for i in xrange(len(pt3[j])-1):
		if ((pn1.x < pt3[j][i][0] and pn2.x > pt3[j][i][0]) or (pn2.x < pt3[j][i][0] and pn1.x > pt3[j][i][0])):
		    return 0    
    return 1
        

def Kruskal(SetOfPoints, type):
    """Kruskal's Algorithm
    Sorts edges by weight, and adds them one at a time to the tree while avoiding cycles
    Takes any set of Point instances and converts to a dictionary via edge crawling 
    Takes the dictionary and iterates through each level to discover neighbors and weights
    Takes list of point index pairs and converts to list of Lines then returns
    """

    for i in xrange(0,len(SetOfPoints)):
        SetOfPoints[i].reset()
    for i in xrange(0,len(SetOfPoints)):
        for j in xrange(i,len(SetOfPoints)):
            if (i != j):
                
		if type == "R":
		    ch = lineinpoly(SetOfPoints[i],SetOfPoints[j])
		    if ch == 1:
			dist = (abs(SetOfPoints[i].x-SetOfPoints[j].x) 
			        + abs(SetOfPoints[i].y - SetOfPoints[j].y))
		    else:
			dist = 100000
		    
		elif type == "G":
		    checker = lineinpoly(SetOfPoints[i],SetOfPoints[j])
		    if checker == 1:
			dist = math.sqrt(pow((SetOfPoints[i].x-SetOfPoints[j].x),2) + 
			pow((SetOfPoints[i].y - SetOfPoints[j].y),2))
		    else:
			dist = 100000
                
                line = Line(SetOfPoints[i], SetOfPoints[j], dist)
                SetOfPoints[i].update(line)
                SetOfPoints[j].update(line)
                          
            else:
                dist = 100000
                line = Line(SetOfPoints[i], SetOfPoints[j], dist)
                SetOfPoints[i].update(line)
            
    G = {}
    for i in xrange(0,len(SetOfPoints)):
        off = 0
        subset = {}
        for j in xrange(0,len(SetOfPoints[i].edges)):
            subset[j] = SetOfPoints[i].edges[j].w
        G[i] = subset

    subtrees = UnionFind()
    tree = []
    for W,u,v in sorted((G[u][v],u,v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append([u,v])
            subtrees.union(u,v)

    MST = []
    for i in xrange(0,len(tree)):
        point1 = SetOfPoints[tree[i][0]]
        point2 = SetOfPoints[tree[i][1]]
        for j in xrange(0,len(point1.edges)):
            if point2 == point1.edges[j].getOther(point1).get():
                point1.MSTupdate(point1.edges[j])
                point2.MSTupdate(point1.edges[j])
                MST.append(point1.edges[j])
    return MST  

def DeltaMST(SetOfPoints, TestPoint, type):
    """DeltaMST 
    Determines the difference in a MST's total weight after adding a point. 
    """

    if type == "R":
	    MST = Kruskal(SetOfPoints, "R")
    else:
	    MST = Kruskal(SetOfPoints, "G")

    cost1 = 0
    for i in xrange(0,len(MST)):
	    cost1 += MST[i].w

    combo = SetOfPoints + [TestPoint]

    if type == "R":
	    MST = Kruskal(combo, "R")
    else:
	    MST = Kruskal(combo, "G")

    cost2 = 0
    for i in xrange(0,len(MST)):
	    cost2 += MST[i].w
    return cost1 - cost2

def pointinpoly(pnt,pt4):
    global count
    global count23
    inner=0
    xcoord = pnt.x*1.0
    ycoord = pnt.y*1.0    
    for j in xrange(len(pt4)-1):
        if((ycoord > pt4[j][1] and ycoord <= pt4[j+1][1]) or (ycoord > pt4[j+1][1] and ycoord <= pt4[j][1])):
            if((pt4[j][0] + (pt4[j+1][0]-pt4[j][0])*(ycoord-pt4[j][1])*1.00/(pt4[j+1][1]-pt4[j][1]))<xcoord):
                inner += 1
            
    if(inner%2==0):
        #canvas.create_oval(xcoord-5,ycoord-5,xcoord+5,ycoord+5,outline="black",fill="green",width=1)
        return 0
    else:
        #canvas.create_oval(xcoord-5,ycoord-5,xcoord+5,ycoord+5,outline="black",fill="red",width=1)
        return 1

def HananPoints(SetOfPoints):
    """HananPoints
    Produces a set of HananPoints of type Points
    """
    global pt3
    totalSet = SetOfPoints
    SomePoints = []
    HanPoints = []
    for i in xrange(0,len(totalSet)):
	for j in xrange(i,len(totalSet)):
		if i != j:
			HanPoints.append(Point(totalSet[i].x, totalSet[j].y))
			HanPoints.append(Point(totalSet[j].x, totalSet[i].y))
    
    for poi in HanPoints:
	seq = 0
	for i in xrange(o_count):
	    if pointinpoly(poi,pt3[i]) == 1:
		break
	    else:
		seq += 1
	if seq == o_count:
	    SomePoints.append(poi)
    """for i in xrange(0,len(SomePoints)):
		canvas.create_oval(SomePoints[i].x-5,SomePoints[i].y-5,
	                SomePoints[i].x+5,SomePoints[i].y+5, outline="black", fill="pink", width=1)
    """
    return SomePoints 

def BrutePoints(SetOfPoints):
    """BrutePoints
    Produces points with spacing 10 between x values and y values between maximal and minimal 
    existing points.
    This could use some work...
    """
    global pt3
    if SetOfPoints != []:
        WholeSet = []
        xmax = (max(SetOfPoints,key=lambda x: x.x)).x
        xmin = (min(SetOfPoints,key=lambda x: x.x)).x
        ymax = (max(SetOfPoints,key=lambda x: x.y)).y
        ymin = (min(SetOfPoints,key=lambda x: x.y)).y
        setup = 0
        rangex = range(xmin,xmax)
        rangey = range(ymin,ymax)
        for i in rangex[::10]:
            for j in rangey[::10]:
                WholeSet.append(Point(i,j))
            setup += 1
        #print WholeSet
        SomePoints = []
        SomePointstemp = []
        for i in xrange(len(SetOfPoints)):
            a=SetOfPoints[i].x
            b=SetOfPoints[i].y
            SomePointstemp.append([])
            SomePointstemp[i].append(a)
            SomePointstemp[i].append(b)
            
        point_collection = geometry.MultiPoint(list(SomePointstemp))
        SomePoints2 = point_collection.convex_hull
        SomePoints3 = np.array(SomePoints2.exterior)
        for poi in WholeSet:
            seq = 0
	    for i in xrange(o_count):
		if pointinpoly(poi,pt3[i]) == 1:
		    break
		else:
		    seq += 1
	    if seq == o_count:
		SomePoints.append(poi)
        return SomePoints
    else:
        return []
    

    
def computeRSMT():
    """computeRSMT
    Computes the Rectilinear Steiner Minimum Spanning Tree
    Uses HananPoints as a candidate set of points for possible steiner points.
    DeltaMST is used to determine which points are beneficial to the final tree.
    Any point with less than two degree value (two or fewer edges) is not helpful and is removed.
    All final points are printed to the canvas.
    """
    retain_points()
    global RSMT
    global min_xcoord
    global max_xcoord
    global min_ycoord
    global max_ycoord    
    time1 = time.time()
    if RSMT == []:
	    global RectSteinerPoints
	    del RectSteinerPoints[:]
	    Candidate_Set = [0]
	    """pt = Point(169,122)
	    OriginalPoints.append(pt)
	    pt = Point(306,115)
	    OriginalPoints.append(pt)
	    pt = Point(176,250)
	    OriginalPoints.append(pt)
	    pt = Point(315,268)
	    OriginalPoints.append(pt)
	    pt = Point(375,168)
	    OriginalPoints.append(pt)		
	    pt = Point(469,368)
	    OriginalPoints.append(pt)
	    pt = Point(69,268)
	    OriginalPoints.append(pt)
	    """
	    for i in xrange(o_count):
		min_coord = np.amin(pt3[i],axis=0)
		max_coord = np.amax(pt3[i],axis=0)
		min_xcoord.append(min_coord[0])
		min_ycoord.append(min_coord[1])
		max_xcoord.append(max_coord[0])
		max_ycoord.append(max_coord[1])

	    maxPoint = Point(1,1)
	    while (maxPoint.x != 0 and maxPoint.y != 0):
		maxPoint = Point(0,0)
		cost = 0
		for x in HananPoints(OriginalPoints + RectSteinerPoints):
		    if DeltaMST(OriginalPoints + RectSteinerPoints,x, "R") > 0:
			DeltaCost = DeltaMST(OriginalPoints + RectSteinerPoints, x, "R")
			if DeltaCost > cost:
				maxPoint = x
				cost = DeltaCost			
			
		if (maxPoint.x != 0 and maxPoint.y != 0):
			RectSteinerPoints.append(maxPoint)
		for pt in RectSteinerPoints:
			if pt.deg <= 2:
				RectSteinerPoints.remove(pt)
			else:
				pass

	    RSMT = Kruskal(OriginalPoints+RectSteinerPoints, "R")

    RSMTminDist = 0
    for i in xrange(0,len(RSMT)):
	    decision = random.randint(0,1)
	    RSMTminDist += RSMT[i].w
	    apoint = Point(RSMT[i].points[0].get().x, RSMT[i].points[0].get().y)
	    bpoint = Point(RSMT[i].points[1].get().x, RSMT[i].points[1].get().y)
	    point1 = Point(RSMT[i].points[0].get().x,RSMT[i].points[1].get().y)
	    point2 = Point(RSMT[i].points[1].get().x,RSMT[i].points[0].get().y)
	    """canvas.create_oval(point1.x-5,point1.y-5,
			        point1.x+5,point1.y+5, outline="black", fill="pink", width=1)
	    canvas.create_oval(point2.x-5,point2.y-5,
			                    point2.x+5,point2.y+5, outline="black", fill="blue", width=1)	
	    """
	    if (lineinpoly(point1,apoint) == 1) and (lineinpoly(point1,bpoint) == 1):
		decision = 0
		print "point 1"
	    elif (lineinpoly(point2,apoint) == 1) and (lineinpoly(point2,bpoint) == 1):
		decision = 1
		print "point 2"
	    else:
		RSMTminDist = 100000
	    
	    
	    if decision == 0:
		    
		    canvas.create_line(RSMT[i].points[0].get().x, RSMT[i].points[0].get().y, 
		            RSMT[i].points[0].get().x, RSMT[i].points[1].get().y, width=2)
		    canvas.create_line(RSMT[i].points[0].get().x, RSMT[i].points[1].get().y, 
		            RSMT[i].points[1].get().x, RSMT[i].points[1].get().y, width=2)
	    else:
		    
		    canvas.create_line(RSMT[i].points[0].get().x, RSMT[i].points[0].get().y, 
		            RSMT[i].points[1].get().x, RSMT[i].points[0].get().y, width=2)
		    canvas.create_line(RSMT[i].points[1].get().x, RSMT[i].points[0].get().y, 
		            RSMT[i].points[1].get().x, RSMT[i].points[1].get().y, width=2)

    for i in xrange(0,len(RectSteinerPoints)):
	    canvas.create_oval(RectSteinerPoints[i].x-5,RectSteinerPoints[i].y-5,
	            RectSteinerPoints[i].x+5,RectSteinerPoints[i].y+5, outline="black", fill="black", width=1)

    for i in xrange(0,len(OriginalPoints)):
	    canvas.create_oval(OriginalPoints[i].x-5,OriginalPoints[i].y-5,
	            OriginalPoints[i].x+5,OriginalPoints[i].y+5, outline="black", fill="white", width=1)

    if RSMTminDist < 100000:
	RSMTtext.set(str(RSMTminDist))
    else:
	RSMTtext.set("Unable to Connect Points")
    time2 = time.time()
    total_time = time2-time1;
    TimetextRect.set(str(round(total_time, 2)))   

def computeGSMT(): 
    """computeGSMT
    Computes the Euclidean Graphical Steiner Minimum Spanning Tree
    Uses BrutePoints as a candidate set of points for possible steiner points. (Approximation factor of <= 2)
    DeltaMST is used to determine which points are beneficial to the final tree.
    Any point with less than two degree value (two or fewer edges) is not helpful and is removed.
    All final points are printed to the canvas.
    """
    retain_points()
    global GSMT
    global Obstacle_pt
    global count23
    time1 = time.time()
    if GSMT == []:
        global GraphSteinerPoints
        del GraphSteinerPoints[:]
        Candidate_Set = [0]
        """pt = Point(488,254)
	OriginalPoints.append(pt)
	pt = Point(339,236)
	OriginalPoints.append(pt)
	pt = Point(302,441)
	OriginalPoints.append(pt)
	pt = Point(442,376)
	OriginalPoints.append(pt)
	pt = Point(518,412)
	OriginalPoints.append(pt)		
	pt = Point(614,325)
	OriginalPoints.append(pt)
	pt = Point(280,264)
	OriginalPoints.append(pt)
	pt = Point(225,329)
	OriginalPoints.append(pt)		
	pt = Point(415,491)
	OriginalPoints.append(pt)
	pt = Point(580,223)
	OriginalPoints.append(pt)	
	"""
	
	check = BrutePoints(OriginalPoints)			
	maxPoint = Point(1,1)
	while (maxPoint.x != 0 and maxPoint.y != 0):
		maxPoint = Point(0,0)
		cost = 0
		for x in BrutePoints(OriginalPoints):
			DeltaCost = DeltaMST(OriginalPoints + GraphSteinerPoints, x,"G")
			if DeltaCost > cost:
				maxPoint = x
				cost = DeltaCost
				print cost
		if (maxPoint.x != 0 and maxPoint.y != 0):
		    GraphSteinerPoints.append(maxPoint)
		for pt in GraphSteinerPoints:
		    if pt.deg <= 2:
			GraphSteinerPoints.remove(pt)
		    else:
			pass


        GSMT = Kruskal(OriginalPoints+GraphSteinerPoints, "G")

    GSMTminDist = 0
    for i in xrange(0,len(GSMT)):
        GSMTminDist += GSMT[i].w
        canvas.create_line(GSMT[i].points[0].get().x, GSMT[i].points[0].get().y, 
            GSMT[i].points[1].get().x, GSMT[i].points[1].get().y, width=2)
        
    for i in xrange(0,len(GraphSteinerPoints)):
        canvas.create_oval(GraphSteinerPoints[i].x-5,GraphSteinerPoints[i].y-5,
            GraphSteinerPoints[i].x+5,GraphSteinerPoints[i].y+5, outline="black", fill="black", width=1)

    for i in xrange(0,len(OriginalPoints)):
        canvas.create_oval(OriginalPoints[i].x-5,OriginalPoints[i].y-5,
            OriginalPoints[i].x+5,OriginalPoints[i].y+5, outline="black", fill="white", width=1)
          
    if GSMTminDist < 100000:
	GSMTtext.set(str(round(GSMTminDist, 2)))
    else:
	GSMTtext.set("Unable to Connect Points")
    time2 = time.time()
    total_time = time2-time1;
    Timetext.set(str(round(total_time, 2)))
    

def clear():
    """clear
    Cleans the global lists and canvas points and text.
    """
    global OriginalPoints
    del OriginalPoints[:]
    global RectSteinerPoints
    del RectSteinerPoints[:]    
    global GraphSteinerPoints
    del GraphSteinerPoints[:]
    global pt
    pt = [[0 for i in xrange(2)] for i in xrange(1000)]
    global count
    count = 0
    global flag
    flag = 0
    global o_count
    o_count = 0
    global pt3
    pt3 = []
    global RSMT
    del RSMT[:]    
    global GSMT
    del GSMT[:]
    RSMTtext.set("-----")
    GSMTtext.set("-----")
    Timetext.set("-----")
    TimetextRect.set("-----")
    Obstext.set("-----")
    ObsTime.set("-----")
    canvas.delete("all")
    length = len(pt2) 
    
    
def clear_Obs():
    """clear
    Cleans the global lists and canvas points and text.
    """
    global OriginalPoints
    del OriginalPoints[:]
    global RectSteinerPoints
    del RectSteinerPoints[:]    
    global GraphSteinerPoints
    del GraphSteinerPoints[:]
    global pt
    pt = [[0 for i in xrange(2)] for i in xrange(1000)]
    global count
    count = 0
    global RSMT
    del RSMT[:]    
    global GSMT
    del GSMT[:]
    RSMTtext.set("-----")
    GSMTtext.set("-----")
    Timetext.set("-----")
    TimetextRect.set("-----")
    canvas.delete("all") 
    for i in range(len(pt3)):
	for j in range(len(pt3[i])-1):	    
	    canvas.create_oval(pt3[i][j][0]-5,pt3[i][j][1]-5,pt3[i][j][0]+5,pt3[i][j][1]+5,outline="black", fill="white", width=1)   
	    line=canvas.create_line(pt3[i][j][0],pt3[i][j][1],pt3[i][j+1][0],pt3[i][j+1][1])
	    
def retain_points():
    """clear
    Cleans the global lists and canvas points and text.
    """
    global RectSteinerPoints
    del RectSteinerPoints[:]    
    global GraphSteinerPoints
    del GraphSteinerPoints[:]
    global pt
    pt = [[0 for i in xrange(2)] for i in xrange(1000)]
    global count
    count = 0
    global RSMT
    del RSMT[:]    
    global GSMT
    del GSMT[:]
    RSMTtext.set("-----")
    GSMTtext.set("-----")
    Timetext.set("-----")
    TimetextRect.set("-----")
    canvas.delete("all") 
    for point in OriginalPoints:
	canvas.create_oval(point.x-5,point.y-5,point.x+5,point.y+5,outline="black", fill="white", width=1)
    for i in range(len(pt3)):
	for j in range(len(pt3[i])-1):	    
	    canvas.create_oval(pt3[i][j][0]-5,pt3[i][j][1]-5,pt3[i][j][0]+5,pt3[i][j][1]+5,outline="black", fill="white", width=1)   
	    line=canvas.create_line(pt3[i][j][0],pt3[i][j][1],pt3[i][j+1][0],pt3[i][j+1][1])
    


master = Canvas(tk)
but_frame = Frame(master)
var = StringVar()
var.set("Distance:")
var2 = StringVar()
var2.set("Time:")
var3 = StringVar()
var3.set("No. of Obstacles:")
var4 = StringVar()
var4.set("Time")
button3 = Button(but_frame, text = "Obstacle", command = computeObstacle)
button3.configure(width=9, activebackground = "blue", relief = RAISED)
button3.pack(side=TOP)
Label(but_frame, textvariable="").pack()
button1 = Button(but_frame, text = "Obstacles Marked", command = completeObstacles)
button1.configure(width=15, activebackground = "blue", relief = RAISED)
button1.pack(side=TOP)
Label(but_frame, textvariable=var3).pack()
Obstext = StringVar()
label1 = Label(but_frame, textvariable=Obstext)
label1.pack()
Label(but_frame, textvariable=var4).pack()
ObsTime = StringVar()
label6 = Label(but_frame, textvariable=ObsTime)
label6.pack()
#Label = (but_frame, textvariable="").pack()
button2 = Button(but_frame, text = "RSMT", command = computeRSMT)
button2.configure(width=9, activebackground = "blue", relief = RAISED)
button2.pack(side=TOP)
Label(but_frame, textvariable=var).pack()
RSMTtext = StringVar()
label2 = Label(but_frame, textvariable=RSMTtext)
label2.pack()
Label(but_frame, textvariable=var2).pack()
TimetextRect = StringVar()
label5 = Label(but_frame, textvariable=TimetextRect)
label5.pack()
Label(but_frame, textvariable="").pack()
button4 = Button(but_frame, text = "GSMT", command = computeGSMT)
button4.configure(width=9, activebackground = "blue", relief = RAISED)
button4.pack(side=TOP)
Label(but_frame, textvariable=var).pack()
GSMTtext = StringVar()
label4 = Label(but_frame, textvariable=GSMTtext)
label4.pack()
Label(but_frame, textvariable=var2).pack()
Timetext = StringVar()
label5 = Label(but_frame, textvariable=Timetext)
label5.pack()
Label(but_frame, textvariable="").pack()
button3 = Button(but_frame, text = "Retain Obstacle", command = clear_Obs)
button3.configure(width=15, activebackground = "blue", relief = RAISED)
button3.pack(side=TOP)
Label(but_frame, textvariable="").pack()
button5 = Button(but_frame, text = "Reset All", command = clear)
button5.configure(width=9, activebackground = "blue", relief = RAISED)
button5.pack(side=TOP)
but_frame.pack(side=RIGHT, expand=0)
canvas = Canvas(master, width = 800, height = 800, bd=2, relief=RIDGE, bg='#F6F5F1')
canvas.bind("<Button-1>", addMousePoint)
canvas.pack(expand=0)
master.pack(expand=0)

RSMTtext.set("-----")
GSMTtext.set("-----")
Timetext.set("-----")
TimetextRect.set("-----")
Obstext.set("-----")
ObsTime.set("-----")

tk.mainloop()