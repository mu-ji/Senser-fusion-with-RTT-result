import numpy as np
from matplotlib import pyplot as plt

def estimate_point_position(x1, y1, x2, y2, x3, y3, r1, r2, r3):
    grid_size = 0.01
    max_x = max(x1+r1,x2+r2,x3+r3)
    min_x = min(x1-r1,x2-r2,x3-r3)
    max_y = max(y1+r1,y2+r2,y3+r3)
    min_y = min(y1-r1,y2-r2,y3-r3)
    x_range = np.arange(min_x,max_x,grid_size)
    y_range = np.arange(min_y,max_y,grid_size)
    inner_list = []
    est_x = 0
    est_y = 0
    for i in x_range:
        for j in y_range:
            if (i-x1)**2+(j-y1)**2 < r1**2 and (i-x2)**2+(j-y2)**2 < r2**2 and (i-x3)**2+(j-y3)**2 < r3**2:
                inner_list.append([i,j])
    
    for i in inner_list:
        est_x = est_x + i[0]*(1/len(inner_list))
        est_y = est_y + i[1]*(1/len(inner_list))

    return est_x,est_y


x1, y1 = 0, 0
x2, y2 = 5, 0
x3, y3 = 0, 5
r1, r2, r3 = 4, 2, 6

estimated_x, estimated_y = estimate_point_position(x1, y1, x2, y2, x3, y3, r1, r2, r3)
print("Estimated point position: ({}, {})".format(estimated_x, estimated_y))

def draw_circle(x1,y1,r):
    theta = np.linspace(0, 2*np.pi, 100)  # 角度从0到2π均匀分布
    x = x1 + r * np.cos(theta)  # x坐标
    y = y1 + r * np.sin(theta)  # y坐标
    return x,y

circle1_x,circle1_y = draw_circle(x1,y1,r1)
circle2_x,circle2_y = draw_circle(x2,y2,r2)
circle3_x,circle3_y = draw_circle(x3,y3,r3)

plt.figure()
plt.plot(circle1_x,circle1_y,c='r')
plt.plot(circle2_x,circle2_y,c='b')
plt.plot(circle3_x,circle3_y,c='g')
plt.scatter(x1,y1,c='r')
plt.scatter(x2,y2,c='b')
plt.scatter(x3,y3,c='g')
plt.scatter(estimated_x,estimated_y,c='y')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.axis("equal")
plt.show()