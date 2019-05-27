import cv2

src = cv2.imread("../image/7.jpg")
img = cv2.imread("result/0007.png")
src = cv2.pyrDown(src)

rows, cols, channels = img.shape
roi = src[:rows, :cols]

img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
mask2 = cv2.bitwise_not(mask)

bg = cv2.bitwise_and(roi, roi, mask=mask)
fg = cv2.bitwise_and(img, img, mask=mask2)

dst = cv2.add(bg, fg)
img[:rows, :cols] = dst

# dst = cv2.pyrDown(dst)
# dst = cv2.pyrDown(dst)

cv2.imshow("image", dst)
cv2.waitKey(0)
