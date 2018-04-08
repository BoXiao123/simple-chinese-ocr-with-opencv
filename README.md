# simple-chinese-ocr-with-opencv

Opencv is an important library for digital image processing. This repository implemented a simple chinese ocr: recognizing book name, with opencv.

![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/1.png)

## Main pipelines

First of all, we do the perspective transform to get the bird view image. And get the ROI.
		def perspective_image(img,
                      src,
                      dst= np.float32([
                            [0,0],
                            [600,0],
                            [0,1200],
                            [600,1200]]),):
		img, Minv = warp(img, src, dst)
		return img, Minv
		
		def warp(img, src, dst):
		M = cv2.getPerspectiveTransform(src, dst)
		Minv = cv2.getPerspectiveTransform(dst, src)
		warped = cv2.warpPerspective(img, M, img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
		return warped, Minv
		
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/1.jpg)
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/2.jpg)

Then we need to transfer BGR to HSV in order to get the black color regions.
		hsv = cv2.cvtColor(perspective, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0,0,0])
        upper_black = np.array([200,255,54])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        res = cv2.bitwise_and(perspective,perspective, mask= mask)
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/3.jpg)

Next, we do erode and dilate to the image.
		def preprocessimage(img,n):
			kernel=np.uint8(np.zeros((n,n)))
			for x in range(n):
				kernel[x,2]=1
				kernel[2,x]=1
			eroded=cv2.erode(img,kernel)
			dilated = cv2.dilate(img, kernel)
			result = cv2.absdiff(dilated, eroded)
			return result
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/4.jpg)

Right now, we can find contours in this image and the largest contour is the name of book.
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/5.jpg)
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/6.jpg)

Change it to the binary image and then calculate the pixel value in the vertical axis you can do segmentation.
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/7.jpg)
![](https://github.com/BoXiao123/simple-chinese-ocr-with-opencv/raw/master/imgs/2.png)

Finally, train a classifier we can finish this simple ocr.

