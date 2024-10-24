import cv2
import numpy as np
# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

# 标准霍夫线变换
def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    gray = 255 - gray
    # cv2.imwrite("/media/liuxz/3EA0B4CEA0B48E41/shiyan/edge.bmp", edges)
    lines = cv2.HoughLines(gray, 1, np.pi /36, 365)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)   # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))   # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))   # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return  image

# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 100, minLineLength=200, maxLineGap=150)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image

if __name__ == "__main__":
    img = cv2.imread("/media/liuxz/3EA0B4CEA0B48E41/shiyan/001_000001_003_2021-03-26T0913129636621.bmp")
    image_biaozhun = line_detection_demo(img)
    cv2.imwrite("/media/liuxz/3EA0B4CEA0B48E41/shiyan/image_biaozhun.bmp", image_biaozhun)

    # image_tongji = line_detect_possible_demo(img)
    # cv2.imwrite("/media/liuxz/3EA0B4CEA0B48E41/shiyan/image_tongji.bmp", image_tongji)