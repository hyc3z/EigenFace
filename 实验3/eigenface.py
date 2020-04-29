import cv2
import numpy as np
import os
import sklearn

model = cv2.face.EigenFaceRecognizer_create()

train_images = []
train_label = []
test_images = []
test_label = []

def start():
    s1_mat = cv2.imread("att_faces/att_faces/s1/1.pgm", 0).flatten()
    pic_dir = "att_faces/att_faces"
    shape:tuple
    for i in os.listdir(pic_dir):
        if os.path.isdir(os.path.join(pic_dir, i)):
            people_dir = os.path.join(pic_dir,i)
            for image in os.listdir(people_dir):
                img = cv2.imread(os.path.join(people_dir, image), 0)

                shape = img.shape
                # print(shape)
                mat_ = np.mat(img).flatten()
                if image[:-4] != "10":
                    train_images.append(mat_)
                    train_label.append(int(i[1:]))
                else:
                    test_images.append(mat_)
                    test_label.append(int(i[1:]))

    model.train(train_images, np.array(train_label))
    # results = []
    # for i in test_images:
    #     result = model.predict(i)
    #     print(result)
    #     results.append(result[0])
    # print("预测结果：\n",results)
    # print("实际结果:\n",test_label)
    eigenvalues = model.getEigenValues()
    # print(eigenvalues)
    eigenvectors = model.getEigenVectors()
    print(eigenvectors.shape)
    mean = model.getMean()
    img = np.resize(mean, shape)
    img2 = img.astype(np.uint8)
    print(img2)
    print(img2.shape)
    s1_im_vector = s1_mat - mean
    # cv2.imshow("mean", img2)
    # cv2.waitKey(0)
    output = mean
    for i in range(0, 360, 1):
        weight = np.dot(s1_im_vector, eigenvectors[:, i])
        output = output + eigenvectors[:, i]* weight
    resized_output = np.resize(output, shape)
    resized_output_2 = resized_output.astype(np.uint8)
    cv2.imshow("",resized_output_2)
    cv2.waitKey(0)

print(model)

if __name__ == '__main__':
    start()