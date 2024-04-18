import cv2
import numpy as np
import utils


webCamFeed = True
pathImage = "4.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
heightImg = 700
widthImg  = 700
questions = 5
choices = 5
ans = [1,2,0,2,4]

count = 0

while True:
    start_time = cv2.getTickCount()

    # if webCamFeed:
    #     success, img = cap.read()
    # else:
    img = cv2.imread(pathImage)

    # print("Image shape:", img.shape)
    img = cv2.resize(img, (widthImg, heightImg)) # resize 
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # blank image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # gaussian blur
    imgCanny = cv2.Canny(imgBlur,10,70) # canny edge detection

    cv2.imshow("Gaussian Blur ", imgBlur)

    try:
        # find contours
        imgContours = img.copy() 
        imgBigContour = img.copy() 
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find all contours
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # draw detected contours
        rectCon = utils.rectContour(contours) # filter for rectangle contours
        biggestPoints = utils.getCornerPoints(rectCon[0]) # corner points of biggest rectangle
        gradePoints = utils.getCornerPoints(rectCon[1]) # corner points of second biggest rectangle

        
        if biggestPoints.size != 0 and gradePoints.size != 0:

            # biggest rectangle warp
            biggestPoints=utils.reorder(biggestPoints) # reorder
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # draw biggest contour
            pts1 = np.float32(biggestPoints) 
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # get transformation matrix
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # apply warp perspective

            # second biggest rectangle warp
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20) 
            gradePoints = utils.reorder(gradePoints) 
            ptsG1 = np.float32(gradePoints)  
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) 

            # thresholding
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # convert to grayscale
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # apply threshold

            boxes = utils.splitBoxes(imgThresh) # get individual boxes
            cv2.imshow("Split Test ", boxes[3])
            countR = 0
            countC = 0
            myPixelVal = np.zeros((questions,choices)) # to store non-zero values of each box
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices):
                    countC = 0
                    countR += 1

            # find user answers and put them in a list
            myIndex = []
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print("User Answers ",myIndex)

            # compare to find correct answers
            grading = []
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            
            score = (sum(grading)/questions)*100 # final score
            
            # display answers
            utils.showAnswers(imgWarpColored,myIndex,grading,ans) # draw detected answers
            utils.drawGrid(imgWarpColored) 
            imgRawDrawings = np.zeros_like(imgWarpColored) 
            utils.showAnswers(imgRawDrawings, myIndex, grading, ans) 
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # inverse transformation matrix
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) # inverse image wrap

            # display grade
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8)
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3) 
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) 
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) 

            # show answers and score on final image
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)

            imageArray = ([img,imgGray,imgCanny,imgContours],
                          [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
        
    end_time = cv2.getTickCount()
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
    print(f"Time taken for processing frame: {elapsed_time} seconds")

    # labels for display
    labels = [["Original","Gray","Edges","Contours"],
              ["Biggest Contour","Threshold","Warped","Final"]]

    stackedImage = utils.stackImages(imageArray,0.5,labels)
    cv2.imshow("Result", stackedImage)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
        cap.release()

    elif key == ord('q'):
        cap.release()
        break
