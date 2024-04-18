import cv2
import numpy as np
import utils

# Configuration
webCamFeed = True
pathImage = "1.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]

count = 0
myIndex = []
myPixelVal = np.zeros((questions, choices))

while True:

    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    try:
        # FIND ALL CONTOURS
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        rectCon = utils.rectContour(contours)
        biggestPoints = utils.getCornerPoints(rectCon[0])
        gradePoints = utils.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            biggestPoints = utils.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # SECOND BIGGEST RECTANGLE WARPING
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
            gradePoints = utils.reorder(gradePoints)
            ptsG1 = np.float32(gradePoints)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            # DETECT CIRCLES
            imgCircles = img.copy()
            imgCircles = cv2.cvtColor(imgCircles, cv2.COLOR_BGR2GRAY)
            imgCircles = cv2.medianBlur(imgCircles, 5)
            circles = cv2.HoughCircles(imgCircles, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=30, minRadius=5, maxRadius=30)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle_coordinates = []
                for i in circles[0, :]:
                    circle_coordinates.append((i[0], i[1]))
                circle_coordinates.sort(key=lambda x: x[1])

                marked_questions = set()

                for x in range(0, questions):
                    arr = myPixelVal[x]
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])

                grading = []
                for x in range(0, questions):
                    if ans[x] == myIndex[x]:
                        grading.append(1)
                        marked_questions.add(x)
                    elif x not in marked_questions:
                        grading.append(0)

                multiple_marked = False
                for i in range(len(circle_coordinates) - 1):
                    if circle_coordinates[i][1] == circle_coordinates[i + 1][1]:
                        multiple_marked = True
                        break

                if multiple_marked:
                    print("Multiple circles marked for one question. Grading marked questions as 0.")
                    for marked_question in marked_questions:
                        grading[marked_question] = 0

                score = (sum(grading) / len(grading)) * 100  # FINAL GRADE

                # DISPLAY GRADE
                imgRawGrade = np.zeros_like(imgFinal, np.uint8)
                cv2.putText(imgRawGrade, str(int(score)) + "%", (int(imgRawGrade.shape[1] / 2) - 70, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)  # ADD THE GRADE TO NEW IMAGE

                # SHOW ANSWERS AND GRADE ON FINAL IMAGE
                imgFinal = cv2.addWeighted(imgFinal, 1, imgRawGrade, 1, 0)

                # IMAGE ARRAY FOR DISPLAY
                imageArray = ([img, imgGray, imgCanny, imgContours],
                              [imgBigContour, imgThresh, imgWarpColored, imgFinal])
                cv2.imshow("Final Result", imgFinal)

    except Exception as e:
        print("Exception:", e)
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    labels = [["Original", "Gray", "Edges", "Contours"],
              ["Biggest Contour", "Threshold", "Warped", "Final"]]

    stackedImage = utils.stackImages(imageArray, 0.5, labels)
    cv2.imshow("Result", stackedImage)

    key = cv2.waitKey(1) & 0xFF
    # SAVE IMAGE WHEN 's' key is pressed
    if key == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgFinal)
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

cv2.destroyAllWindows()
