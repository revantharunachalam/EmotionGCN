import numpy as np
import cv2

def draw_border(frame, pt1, pt2, color, thickness, r, d):
    """ Draw a fancy rounded rectangle.
        frame: cv2 frame
        pt1: point 1; Starting point 
        pt2: point 2; Ending point
        color: color BGR
        thickness: The thickness of the fancy rectangle
        r: Radius of ellipse
        d: Diameter of ellipse

        USAGE: draw_border(img, (10,10), (100, 100), (127,255,255), 4, 5, 10)
    """    
    x1,y1 = pt1
    x2,y2 = pt2
    
    # Top left
    cv2.line(frame, (x1+r,y1),(x1+r+d,y1),color,thickness)
    cv2.line(frame, (x1,y1+r),(x1,y1+r+d),color,thickness)
    cv2.ellipse(frame,(x1+r,y1+r),(r,r),180,0,90,color,thickness)

    # Top right
    cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
             
    # Bottom left
    cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
                              
    # Bottom right
    cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
