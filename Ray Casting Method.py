#Ray Casting Method
def point_in_poly(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

polygon = [(0,10),(10,10),(10,0),(0,0)]

point_x = 0
point_y = 0

## Call the function with the points and the polygon
print point_in_poly(point_x,point_y,polygon)

from datetime import datetime
start=datetime.now()

#Statements

print datetime.now()-start

