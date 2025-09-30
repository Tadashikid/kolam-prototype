import cv2 
import numpy as np
import turtle, time

# ---------------------------
# Load image and preprocess
# ---------------------------
img = cv2.imread("kolam2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to extract white Kolam lines
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# ---------------------------
# Detect dots 
# ---------------------------
def detect_dots(img, gray, thresh):
    # Blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 200
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(thresh)

    dots = []
    if len(keypoints) > 0:  # use Blob result if good enough
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            dots.append((x, y, r))
    else:
        # HoughCircles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=15,
                                   param1=50,
                                   param2=10,
                                   minRadius=3,
                                   maxRadius=12)
        if circles is not None:
            dots = [(x, y, r) for (x, y, r) in np.uint16(np.around(circles[0,:]))]

    return dots

dots = detect_dots(img, gray, thresh)

# ---------------------------
# Detect Kolam lines
# ---------------------------
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# ---------------------------
# Show CV windows
# ---------------------------
dot_img = img.copy()
for (x, y, r) in dots:
    cv2.circle(dot_img, (x, y), r, (0, 0, 255), 2)
    cv2.circle(dot_img, (x, y), 2, (255, 0, 0), 3)

line_img = np.zeros_like(img)
cv2.drawContours(line_img, contours, -1, (0,255,0), 1)

cv2.imshow("Original", img)
cv2.waitKey(0); cv2.destroyAllWindows()

cv2.imshow("Thresholded", thresh)
cv2.waitKey(0); cv2.destroyAllWindows()

cv2.imshow("Detected Dots", dot_img)
cv2.waitKey(0); cv2.destroyAllWindows()

cv2.imshow("Detected Lines", line_img)
cv2.waitKey(0); cv2.destroyAllWindows()

# ---------------------------
# Turtle window startup
# ---------------------------
screen = turtle.Screen()
screen.setup(width=800, height=800)
screen.title("Kolam Design with Dots")
t = turtle.Turtle()
t.hideturtle()
t.speed(0)
t.pensize(4)
turtle.tracer(0,0)

h, w = gray.shape

def to_turtle_coords(x, y, img_w, img_h, screen_w=800, screen_h=800):
    scale_x = screen_w / img_w
    scale_y = screen_h / img_h
    tx = x * scale_x - screen_w//2
    ty = screen_h//2 - y * scale_y
    return tx, ty

# ---------------------------
# Draw dots
# ---------------------------
for (x, y, r) in dots:
    tx, ty = to_turtle_coords(x, y, w, h)
    t.penup()
    t.goto(tx, ty)
    t.pendown()
    t.dot(8, "red")
    turtle.update()
    time.sleep(0.001)

# ---------------------------
# Draw lines
# ---------------------------
for cnt in contours:
    if len(cnt) < 2:
        continue
    pts = cnt[:,0,:]
    start = to_turtle_coords(pts[0,0], pts[0,1], w, h)
    t.penup()
    t.goto(start)
    t.pendown()
    for pt in pts[1:]:
        x, y = to_turtle_coords(pt[0], pt[1], w, h)
        t.goto(x,y)
        turtle.update()
        time.sleep(0.00002)

turtle.done()

