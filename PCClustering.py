import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# from sklearn.cluster import estimate_bandwidth,MeanShift
#Load data_pre
def PCClu(fileName):
    pcd = o3d.io.read_point_cloud(fileName)
    temp=np.asarray(pcd.points)
    #o3d.visualization.draw_geometries([pcd])

    downpcd = pcd.voxel_down_sample(voxel_size=0.15)#0.02
    #o3d.visualization.draw_geometries([downpcd])

    #Cropping the image to only left ROI
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-15, -5, -2),#test min_bound=(-11,-3,-1)
        max_bound=(30, 7, 1))
    croppcd = downpcd.crop(bbox)
    #o3d.visualization.draw_geometries([croppcd])

    #Getting the roof points
    roof_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(-1.5, -1.7, -1.0),# test min_bound=(-1.5,-2,-1.2)
        max_bound=(2.6, 1.7, -0.4)   # test max_bound=(2.6,1.7,0)
    )
    roofpcd = croppcd.crop(roof_bbox)
    #o3d.visualization.draw_geometries([roofpcd])

    croppcd_ponints = np.asarray(croppcd.points)
    roofpcd_points = np.asarray(roofpcd.points)

    indices=[]
    for roof_element in roofpcd_points:
        array_comparison=np.equal(croppcd_ponints,roof_element)
        for index,array in enumerate(array_comparison):
            if np.sum(np.logical_and(array,[True,True,True]))==3:
                indices.append(index)

    #Extracting the points that does not belong to the roof
    regionpcd=croppcd.select_by_index(indices,invert=True)
    #o3d.visualization.draw_geometries([regionpcd])

    #Segmententation of geometric primitives from point clouds using RANSAC
    #distance_threshold: the parameter of threshhold
    #ransac_n the number of points of each iteration
    plane_model,inliers=regionpcd.segment_plane(distance_threshold=0.2,ransac_n=3,num_iterations=100)#0.2 3

    #The inner cloud
    inlier_cloud=regionpcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0,0,0])

    #The outer cloud
    outlier_cloud=regionpcd.select_by_index(inliers,invert=True)
    #o3d.visualization.draw_geometries([inlier_cloud])
    #o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels=np.array(outlier_cloud.cluster_dbscan(eps=0.6 ,min_points=10,print_progress=False))#True
    max_label=labels.max()
    colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
    colors[labels<0]=0
    outlier_cloud.colors=o3d.utility.Vector3dVector(colors[:,:3])
    outlier_cloud.paint_uniform_color([0, 1, 0])
    #o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud])

    bounding_boxes=[]
    bounding_boxes_array=[]
    bounding_boxes_array_test = []
    temp=np.unique(labels)
    for cluster_number in list(np.unique(labels))[1:]:
        cluster_indices=np.where(labels==cluster_number)
        cluster_cloud_points=outlier_cloud.select_by_index(cluster_indices[0].tolist())
        outlier_cloud.select_by_index(cluster_indices[0].tolist()).paint_uniform_color([np.random.uniform(),np.random.uniform(),np.random.uniform()])
        pointsArray=np.asarray(cluster_cloud_points.points)
        if len(pointsArray)<=50 :#or len(pointsArray)>=700:
             continue
        minX, maxX = np.min(pointsArray[:, 0]), np.max(pointsArray[:, 0])
        minY, maxY = np.min(pointsArray[:, 1]), np.max(pointsArray[:, 1])
        minZ, maxZ = np.min(pointsArray[:, 2]), np.max(pointsArray[:, 2])
        # print(f'min value in the array of the points：{np.min(pointsArray)},max value in the array of the points：{np.max(pointsArray)} ')
        # print(f'minX:{minX}, maxX:{maxX}\n minY:{minY}, maxY:{maxY}\n minZ:{minZ}, maxZ:{maxZ}\n')
        object_bbox=cluster_cloud_points.get_axis_aligned_bounding_box()#返回一个轴对齐的几何体边界框
        bounding_boxes.append(object_bbox)
        bounding_boxes_array.append([[minX,minY,minZ],[maxX,maxY,maxZ]])
        #bounding_boxes_array.append([list(object_bbox.min_bound),list(object_bbox.max_bound)])#bounding box的左下和右上坐标，可以确定bounding box形状
    return inlier_cloud,outlier_cloud,bounding_boxes,bounding_boxes_array

def calBox(bounding_box_array):
    allBox=[]
    for i in range(len(bounding_box_array)):
        min_bound,max_bound=bounding_box_array[i][0],bounding_box_array[i][1]
        x,y,z=(min_bound[0]+max_bound[0])/2,(min_bound[1]+max_bound[1])/2,(min_bound[2]+max_bound[2])/2
        h,w,l,=(max_bound[0]-min_bound[0]),(max_bound[1]-min_bound[1]),(max_bound[2]-min_bound[2])
        allBox.append([2, 0.0000, 0, 0.0000, 0.0000, 0.0000, 0.0000 ,0.0000,l,w,h,x,y,z,0,0])#max(1,min(i,5))
    return allBox
if __name__ == '__main__':
    fileName= r"data/predictions/0000000007.pcd"
    inlier_cloud, outlier_cloud, bounding_boxes,bounding_box_array=PCClu(fileName)
    allBox=calBox(bounding_box_array)
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud,*bounding_boxes])