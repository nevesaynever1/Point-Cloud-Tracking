from __future__ import print_function
#import matplotlib; matplotlib.use('Agg')
import os.path, copy, numpy as np, time, sys
from numba import jit
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils import load_list_from_folder, fileparts, mkdir_if_missing
from scipy.spatial import ConvexHull
import open3d as o3d
import numpy as np
import time
import os
from alfred.fusion.common import draw_3d_box, compute_3d_box_lidar_coords
from PCClustering import PCClu,calBox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from alfred.fusion.kitti_fusion import load_pc_from_file

@jit
def poly_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

@jit        
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

@jit       
def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

@jit       
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and 
        convert it to the 8 corners of the 3D box
        
        Returns:
            corners_3d: (8,3) array in in rect camera coord
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])    

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]
    
    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + bbox3d[0]
    corners_3d[1,:] = corners_3d[1,:] + bbox3d[1]
    corners_3d[2,:] = corners_3d[2,:] + bbox3d[2]
 
    return np.transpose(corners_3d)

class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self, bbox3D, info):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=10, dim_z=7)
    self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0],      # state transition matrix
                          [0,1,0,0,0,0,0,0,1,0],
                          [0,0,1,0,0,0,0,0,0,1],
                          [0,0,0,1,0,0,0,0,0,0],  
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,1]])     
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
                          [0,1,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,0,0,0,0],
                          [0,0,0,1,0,0,0,0,0,0],
                          [0,0,0,0,1,0,0,0,0,0],
                          [0,0,0,0,0,1,0,0,0,0],
                          [0,0,0,0,0,0,1,0,0,0]])

    # with angular velocity
    # self.kf = KalmanFilter(dim_x=11, dim_z=7)       
    # self.kf.F = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
    #                       [0,1,0,0,0,0,0,0,1,0,0],
    #                       [0,0,1,0,0,0,0,0,0,1,0],
    #                       [0,0,0,1,0,0,0,0,0,0,1],  
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0],
    #                       [0,0,0,0,0,0,0,1,0,0,0],
    #                       [0,0,0,0,0,0,0,0,1,0,0],
    #                       [0,0,0,0,0,0,0,0,0,1,0],
    #                       [0,0,0,0,0,0,0,0,0,0,1]])     
    
    # self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
    #                       [0,1,0,0,0,0,0,0,0,0,0],
    #                       [0,0,1,0,0,0,0,0,0,0,0],
    #                       [0,0,0,1,0,0,0,0,0,0,0],
    #                       [0,0,0,0,1,0,0,0,0,0,0],
    #                       [0,0,0,0,0,1,0,0,0,0,0],
    #                       [0,0,0,0,0,0,1,0,0,0,0]])

    # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
    self.kf.P[7:,7:] *= 1000.   # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
    self.kf.P *= 10.
    
    #self.kf.Q[-1,-1] *= 0.01    # process uncertainty
    self.kf.Q[7:,7:] *= 0.01
    self.kf.x[:7] = bbox3D.reshape((7, 1))

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 1           # number of total hits including the first detection
    self.hit_streak = 1     # number of continuing hit considering the first detection
    self.first_continuing_hit = 1
    self.still_first = True
    self.age = 0
    self.info = info        # other info

  def update(self, bbox3D, info): 
    """ 
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1          # number of continuing hit
    if self.still_first:
      self.first_continuing_hit += 1      # number of continuing hit in the fist time
    
    ######################### orientation correction
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    new_theta = bbox3D[3]
    if new_theta >= np.pi: new_theta -= np.pi * 2    # make the theta still in the range
    if new_theta < -np.pi: new_theta += np.pi * 2
    bbox3D[3] = new_theta

    predicted_theta = self.kf.x[3]
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:     # if the angle of two theta is not acute angle
      self.kf.x[3] += np.pi
      if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
      if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
      if new_theta > 0: self.kf.x[3] += np.pi * 2
      else: self.kf.x[3] -= np.pi * 2
    
    ######################### 

    self.kf.update(bbox3D)#根据状态结果修改状态向量x

    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2    # make the theta still in the range
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
    self.info = info

  def predict(self):       
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    self.kf.predict()      
    if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
    if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.still_first = False
    self.time_since_update += 1
    self.history.append(self.kf.x)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return self.kf.x[:7].reshape((7, ))

def associate_detections_to_trackers(detections,trackers,iou_threshold=0.01):     #0.01
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.1):      # ablation study
# def associate_detections_to_trackers(detections,trackers,iou_threshold=0.25):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  detections:  N x 8 x 3
  trackers:    M x 8 x 3

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,8,3),dtype=int)    
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou3d(det,trk)[0] #3维点云交并比           # det: 8 x 3, trk: 8 x 3
  
  # matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm     # deprecated
  row_ind, col_ind = linear_sum_assignment(-iou_matrix)      # hougarian algorithm
  matched_indices = np.stack((row_ind, col_ind), axis=1)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):        # A baseline of 3D multi-object tracking
  def __init__(self, max_age=2, min_hits=3):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0

    # self.reorder = [3, 4, 5, 6, 2, 1, 0]#ori
    self.reorder_back = [6, 5, 4, 0, 1, 2, 3]#ori
    self.reorder = [3, 4, 5, 6, 0, 1, 2]
    self.lwh = [2, 1, 0, 3, 4, 5, 6]
    # self.reorder_back = [5, 4, 6, 0, 1, 2, 3]

  def update(self, dets_all):#输入concate之后的数据就可以
    """
    Params:
      dets_all: dict
        dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
        info: a array of other info for each det
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
    
    # reorder the data_pre to put x,y,z in front to be compatible with the state transition matrix
    # where the constant velocity model is defined in the first three rows of the matrix
                                            # original l,w,h,x,y,z,0,0.6646   self.reorder = [3, 4, 5, 6, 0, 1, 2]
    dets = dets[:, self.reorder]            # reorder the data_pre to [[x,y,z,theta,l,w,h], ...]
    self.frame_count += 1

    trks = np.zeros((len(self.trackers),7))         # N x 7 , #get predicted locations from existing trackers.
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict().reshape((-1, 1)) #shape:(10,1)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))   
    for t in reversed(to_del):
      self.trackers.pop(t)

    dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]#检测的8个点坐标
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []
    trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]#追踪目标的点
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)
    
    #update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if t not in unmatched_trks:
        d = matched[np.where(matched[:,1]==t)[0],0]     # a list of index
        trk.update(dets[d,:][0], info[d, :][0])

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:        # a scalar of index
        trk = KalmanBoxTracker(dets[i,:], info[i, :])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()      # bbox location
        d = d[self.reorder_back]    # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
        temp=((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits))
        if((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
          ret.append(np.concatenate((d, [trk.id+1], trk.info)).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #remove dead tracklet
        if(trk.time_since_update >= self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)      # h,w,l,x,y,z,theta, ID, other info, confidence
    return np.empty((0,15))      


def color_sep(label):
    if label%9 == 1 :
        color = [0.5, 0, 1]
    elif label%9 == 2:
        color = [1, 0, 0]
    elif label%9 == 3:
        color = [1, 1, 0]
    elif label%9 == 4:
        color = [0, 1, 1]
    elif label%9 == 5:
        color = [1, 1, 1]
    elif label%9 == 6:
        color = [0.4, 0.5, 1]
    elif label%9 ==7:
        color = [0.4, 1, 0.4]
    elif label%9 ==8:
        color = [0.4, 0.2, 0.6]
    else:
        color = [max(1,1-label/255),max(1-label/255,0.2), min(0.8,1-label/255)]
    #     color = [np.random.uniform(0,1),np.random.uniform(0,1), np.random.uniform(0,1)]
    return color

def visual_option():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    opt.line_width = 100
    opt.show_coordinate_frame = False
    return vis

def load_Kalman(seq_dets,):
    dets = seq_dets[:, 8:15]#l,w,h,x,y,z,0
    ori_array = seq_dets[:, -1].reshape((-1, 1))
    other_array = seq_dets[:, 0:8]
    additional_info = np.concatenate((ori_array, other_array), axis=1)
    dets_all = {'dets': dets, 'info': additional_info}
    trackers = mot_tracker.update(dets_all)#使用匈牙利匹配进行更新

    return trackers




def draw_points(all_points,pause_time=1):
    fig = plt.figure(figsize=(20, 20))
    ax = Axes3D(fig)
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')
    for point in all_points:
        color = point[1]
        xyz = list(point[0][0])
        ax.scatter3D(xyz[0], xyz[1], xyz[2], color=(int(color[0]), int(color[1]), int(color[2])), )
    # ax.legend()
    plt.draw()
    plt.pause(pause_time)
    plt.close(fig)

def all_class_PC(all_points):
    all_IDs = []
    for point in all_points:
        all_IDs.append(point[2])
    unique_IDs = np.unique(all_IDs)
    all_class = [[] for _ in range(len(unique_IDs))]
    for point in all_points:
        id = int(point[2])
        if id in unique_IDs:
            all_class[id - 1].append([point[0], point[1]])
    return all_class

if __name__ == '__main__':
  if len(sys.argv)!=2:
    print("Usage: python main.py result_sha(e.g., car_3d_det_test)")
    sys.exit(1)
  # ====================================Visual Preparation======================================
  vis = visual_option()
  pcobj = o3d.geometry.PointCloud()
  # =====================================Data Extraction========================================
  result_sha = sys.argv[1]

  seq_file_list, num_seq = load_list_from_folder(os.path.join('data/', result_sha))
  total_time = 0.0
  total_frames = 0
  len_trackers=0
  threshold = []
  mot_tracker = AB3DMOT()
  all_points=[]
  all_line_set=[]
  for number,seq_file in enumerate(seq_file_list):
    _, seq_name, _ = fileparts(seq_file)
    inlier_cloud, outlier_cloud, bounding_boxes, bounding_boxes_array=PCClu(seq_file)
    vis.add_geometry(inlier_cloud, )
    vis.update_geometry(inlier_cloud, )
    vis.add_geometry( outlier_cloud,)
    vis.update_geometry( outlier_cloud,)
    seq_dets=np.asarray(calBox(bounding_boxes_array))

    # ====================================Kalman Filter=========================================
    start_time = time.time()
    trackers = load_Kalman(seq_dets,)# [h,w,l,x,y,z,theta] [trk.id+1], trk.info
    len_trackers+=len(trackers)
    cycle_time = time.time() - start_time
    total_time += cycle_time
    for index, d in enumerate(trackers):
        xyz = np.array([d[3: 6]])
        hwl = np.array([d[: 3]])
        id=np.array(d[7])
        r_y = [d[6]]
        pts3d = compute_3d_box_lidar_coords(xyz, hwl, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                 [4, 5], [5, 6], [6, 7], [7, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        color = color_sep(int(d[7]))
        line_colors = [color for i in range(len(lines))]
        all_points.append([xyz,color,id])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts3d[0])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)
        vis.add_geometry(line_set)
        vis.update_geometry(line_set)
        all_line_set.append(line_set)
    #draw_points(all_points)
    # all_PC_points = all_class_PC(all_points)
    threshold.append(index)
    # print("tracker is {}".format(len(trackers)))
    # print("index is {}".format(index))

    #time.sleep(3)
    #print('current frame:',total_frames)
    total_frames += 1
    vis.poll_events()    #注册更新操作
    vis.update_renderer()#渲染图窗
    vis.remove_geometry(inlier_cloud)
    vis.remove_geometry(outlier_cloud)
    for i in range(len(all_line_set)):
        vis.remove_geometry(all_line_set[i])

    #vis.run()
    print("{} has been visualized".format(seq_name))
print("total time:",total_time)
print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
#draw_points(all_points, pause_time=10)

# total_trackers=len(all_points)
# print('len_trackers:',len_trackers)
# print('total trackers:',total_trackers)
# figPC=plt.figure(figsize=(20,20))
# ax=Axes3D(figPC)
# all_IDs=[]
# for point in all_points:
#     all_IDs.append(point[2])
# unique_IDs=np.unique(all_IDs)
# all_class=[[] for _ in range(len(unique_IDs))]
# for point in all_points:
#     id=int(point[2])
#     #print('int(point[2]) in unique_IDs:',int(point[2]) in unique_IDs)
#     if id in unique_IDs:
#         all_class[id-1].append([point[0],point[1]])
#     #print(point[1]!=color)
#     color=point[1]
#     xyz = list(point[0][0])
#     ax.scatter3D(xyz[0], xyz[1], xyz[2], color=(int(color[0]),int(color[1]),int(color[2])))
# plt.show()
# for each_class in all_class:
#     temp=each_class
#     tempShape=np.shape(temp)
#     each_class_point=np.array(each_class)[:,0]
#     #finalClass=np.reshape(test,(len(test),3))
#     #ax.plot(each_class_point)
# plt.show()