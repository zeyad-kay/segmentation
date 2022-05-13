import numpy as np
import cv2 as cv
import random
import time

class Agglomerative :
    def __init__(self,image_path) -> None:
        self.original_image = cv.imread(image_path)
        self.image = np.array(self.original_image)
        self.points = self.image.reshape(self.image.shape[0] * self.image.shape[1] , 3)
        self.distance_matrix = [[-1]*len(self.points) for i in range(0,len(self.points))]
        self.dindogram = [[i] for i in range(len(self.points))]

    def calculate_initial_matrix(self) :
        for i in range(len(self.points)) :
            for j in range(i,len(self.points)) :
                if(j == i) : 
                    self.distance_matrix[j][i] = -1
                    continue
                self.distance_matrix[j][i] = self.calcualte_distance(self.points[i],self.points[j])

    def calcualte_distance(self,p1,p2) :
        return np.sqrt(np.sum(np.square(np.subtract(p1,p2))))

    def get_minimum_distance(self,matrix) :
        minimum = [1,0]
        for i in range(len(matrix)) :
            for j in range(len(matrix[0])) :
                if((matrix[i][j] == -1)) : continue
                if(matrix[i][j] < matrix[minimum[0]][minimum[1]]) :
                    minimum = [i,j]
        return minimum

    def fit(self,number_of_clusters = 2) :
        if(number_of_clusters > len(self.points)) : raise Exception("You can not set the number of clusters to more than the number of points")
        self.calculate_initial_matrix()
        while len(self.dindogram) != number_of_clusters :
            minimum = self.get_minimum_distance(self.distance_matrix)
            new_cluster = [self.dindogram[minimum[0]],self.dindogram[minimum[1]]]
            flat_new_cluster = [item for sublist in new_cluster for item in sublist]
            self.dindogram.pop(np.max(minimum))
            self.dindogram[np.min(minimum)] = flat_new_cluster
            self.update_matrix(minimum[0],minimum[1])

    def update_matrix(self,index1,index2) :
        maximum_indx = max([index1,index2])
        self.single_link(index1,index2)
        self.distance_matrix.pop(maximum_indx)
        for i in range(len(self.distance_matrix)) :
            self.distance_matrix[i].pop(maximum_indx)
    
    def single_link(self,index1,index2) :
        minimum_indx = min([index1,index2])
        for i in range(len(self.distance_matrix)) :
            if(i == index1) : continue
            if(i == index2) : continue
            if(i < index1) :
                if(i < index2) :
                    distanc_1 = self.distance_matrix[index1][i]
                    distanc_2 = self.distance_matrix[index2][i]
                    m = min([distanc_1,distanc_2])
                    self.distance_matrix[minimum_indx][i] = m
                else :
                    distanc_1 = self.distance_matrix[index1][i]
                    distanc_2 = self.distance_matrix[i][index2]
                    m = min([distanc_1,distanc_2])
                    if(minimum_indx == index2) : self.distance_matrix[i][minimum_indx] = m
                    else : self.distance_matrix[i][minimum_indx] = m
                    
            else :
                if(i < index2) :
                    distanc_1 = self.distance_matrix[i][index1]
                    distanc_2 = self.distance_matrix[index2][i]
                    if(minimum_indx == index1) : self.distance_matrix[i][minimum_indx] = m
                    else : self.distance_matrix[i][minimum_indx] = m
                else :
                    distanc_1 = self.distance_matrix[i][index1]
                    distanc_2 = self.distance_matrix[i][index2]
                    m = min([distanc_1,distanc_2])
                    self.distance_matrix[i][minimum_indx] = m
    
    def image_mask(self) :
        colors = []
        while len(colors) != len(self.dindogram):
            color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            if(color not in colors) : colors.append(color)
        for i in range(len(self.dindogram)) :
            for j in range(len(self.dindogram[i])) :
                indx = self.dindogram[i][j]
                self.points[indx] = colors[i]
        image = self.points.reshape(self.original_image.shape)
        filename = "segmentedImage.png"
        cv.imwrite(filename,image)
        # cv.imshow("segmented image",image)
        # cv.waitKey(0)

############################### usage ###############################
agg = Agglomerative("../images/ss.png")
start_time = time.time()
agg.fit(5)
print("--- %s seconds ---" % (time.time() - start_time))
agg.image_mask()
#####################################################################