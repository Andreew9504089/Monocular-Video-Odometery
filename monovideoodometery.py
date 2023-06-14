import numpy as np
import cv2
import os


class MonoVideoOdometery(object):
    def __init__(self, 
                img_file_path,
                pose_file_path,
                focal_length = 718.8560,
                pp = (607.1928, 185.2157), 
                lk_params=dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)), 
                detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)):
        '''
        Arguments:
            img_file_path {str} -- File path that leads to image sequences
            pose_file_path {str} -- File path that leads to true poses from image sequence
        
        Keyword Arguments:
            focal_length {float} -- Focal length of camera used in image sequence (default: {718.8560})
            pp {tuple} -- Principal point of camera in image sequence (default: {(607.1928, 185.2157)})
            lk_params {dict} -- Parameters for Lucas Kanade optical flow (default: {dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))})
            detector {cv2.FeatureDetector} -- Most types of OpenCV feature detectors (default: {cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)})
        
        Raises:
            ValueError -- Raised when file either file paths are not correct, or img_file_path is not configured correctly
        '''

        self.file_path = img_file_path
        self.detector = detector
        self.lk_params = lk_params
        self.focal = focal_length
        self.pp = pp
        self.R = np.zeros(shape=(3, 3))
        self.t = np.zeros(shape=(3, 3))
        self.K = np.array([[self.focal, 0, pp[0]],
                           [0, self.focal, pp[1]],
                           [0, 0, 1]])
        self.id = 0
        self.n_features = 0
        self.time_cnt = 0

        try:
            if not all([".png" in x for x in os.listdir(img_file_path)]):
                raise ValueError("img_file_path is not correct and does not exclusively png files")
        except Exception as e:
            print(e)
            raise ValueError("The designated img_file_path does not exist, please check the path and try again")

        try:
            with open(pose_file_path) as f:
                self.pose = f.readlines()
        except Exception as e:
            print(e)
            raise ValueError("The pose_file_path is not valid or did not lead to a txt file")

        self.process_frame()


    def hasNextFrame(self):
        '''Used to determine whether there are remaining frames
           in the folder to process
        
        Returns:
            bool -- Boolean value denoting whether there are still 
            frames in the folder to process
        '''

        return self.id < len(os.listdir(self.file_path)) 


    def detect(self, img):
        '''Used to detect features and parse into useable format

        
        Arguments:
            img {np.ndarray} -- Image for which to detect keypoints on
        
        Returns:
            np.array -- A sequence of points in (x, y) coordinate format
            denoting location of detected keypoint
        '''

        p0 = self.detector.detect(img)
        # orb = cv2.ORB_create()
        # p0 = orb.detect(img)
        
        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)


    def visual_odometery(self, post_opt = True, filter_type="Observability"):
        '''
        Used to perform visual odometery. If features fall out of frame
        such that there are less than 2000 features remaining, a new feature
        detection is triggered. 
        '''
        self.time_cnt += 1
        
        if post_opt == True:
            # if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame)

            # Calculate optical flow between frames, st holds status
            # of points from frame to frame
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame, self.current_frame, self.p0, None, **self.lk_params)

            # Save the good points from the optical flow
            self.good_old = self.p0[st == 1]
            self.good_new = self.p1[st == 1]

            self.I_old = cv2.cvtColor(self.old_frame.copy(), cv2.COLOR_GRAY2BGR)
            # for point in self.good_old:
            #     cv2.circle(self.I_old,point.astype(np.int64),2,(0,255,0))
            
            # If the frame is one of first two, we need to initalize
            # our t and R vectors so behavior is different
            if self.id < 2:
                E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
                _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, R = self.R, t = self.t, cameraMatrix=self.K)
            else:
                E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
                _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, R = self.R.copy(), t = self.t.copy(), cameraMatrix=self.K)
        
            if filter_type == "Observability":
                score = []
                for p in self.good_old:
                    pp = np.linalg.inv(self.K) @ np.array([p[0], p[1], 1])
                    skew =np.array([[ 0,    -pp[2],   pp[0]],
                                    [ pp[2],  0,     -pp[1]],
                                    [-pp[1],  pp[0],   0]])

                    H1 = self.K
                    H2 = -self.K @ R @ skew

                    H = np.hstack((H1, H2))
                    M = H @ H.transpose()
                    tr = np.trace(M)
                    score.append(tr)
                            
                score = np.asarray(score)
                m = 1
                mask = (abs(score - np.mean(score)) < m * np.std(score))
                for i in range(mask.shape[0]):
                    if mask[i] == False:
                        if score[i] > np.mean(score):
                            mask[i] = True
                
                p1 = self.good_old[mask == True]
                p2 = self.good_new[mask == True]
                              
                E, _ = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
                _, R, t, _ = cv2.recoverPose(E, p1, p2, R = self.R.copy(), t = self.t.copy(), cameraMatrix=self.K)
            
            if filter_type ==  "Center":
                mask = np.ones(len(self.good_old))
                center = np.asarray((self.I_old.shape[1]/2, self.I_old.shape[0]/2))
                
                for i,p in enumerate(self.good_old):
                    if np.linalg.norm(p - center) < 100:
                        mask[i] = False
                    else:
                        mask[i] = True
                        
                p1 = self.good_old[mask == True]
                p2 = self.good_new[mask == True]
                E, _ = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
                _, R, t, _ = cv2.recoverPose(E, p1, p2, R = self.R.copy(), t = self.t.copy(), cameraMatrix=self.K)       
                        
                self.result = cv2.circle(self.I_old,center.astype(np.int64),5,(255,0,0), 0)
                
            if filter_type ==  "Random":
                mask = (np.random.rand(len(self.good_old)) > 0.1).astype(int)
                
                p1 = self.good_old[mask == True]
                p2 = self.good_new[mask == True]
                E, _ = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
                _, R, t, _ = cv2.recoverPose(E, p1, p2, R = self.R.copy(), t = self.t.copy(), cameraMatrix=self.K)       
                                        
            for i,point in enumerate(self.good_old):
                if mask[i]:
                    self.result = cv2.circle(self.I_old,point.astype(np.int64),2,(0,255,0))
                else:
                    self.result = cv2.circle(self.I_old,point.astype(np.int64),2,(0,0,255))
            
            print(len(p1) - len(self.good_old))
            
            if self.id < 2:
                self.R = R
                self.t = t
                
            else:
                absolute_scale = self.get_absolute_scale()
                if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                    self.t = self.t + absolute_scale*self.R.T.dot(t)
                    self.R = R.dot(self.R)
                    
            # Save the total number of good features
            self.n_features = p2.shape[0]
        
        else:
            # if self.n_features < 2000:
            self.p0 = self.detect(self.old_frame.copy())

            # Calculate optical flow between frames, st holds status
            # of points from frame to frame
            self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_frame.copy(), self.current_frame, self.p0, None, **self.lk_params)

            # Save the good points from the optical flow
            self.good_old = self.p0[st == 1]
            self.good_new = self.p1[st == 1]

            self.I_old = self.old_frame.copy()
            for point in self.good_old:
                self.result = cv2.circle(self.I_old,point.astype(np.int64),2,(255,0,0))
            # If the frame is one of first two, we need to initalize
            # our t and R vectors so behavior is different
            if self.id < 2:
                E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
                _, self.R, self.t, _ = cv2.recoverPose(E, self.good_old, self.good_new, R = self.R, t = self.t, cameraMatrix=self.K)
            else:
                E, _ = cv2.findEssentialMat(self.good_new, self.good_old, self.focal, self.pp, cv2.RANSAC, 0.999, 1.0, None)
                _, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, R = self.R.copy(), t = self.t.copy(), cameraMatrix=self.K)
                absolute_scale = self.get_absolute_scale()
                if (absolute_scale > 0.1 and abs(t[2][0]) > abs(t[0][0]) and abs(t[2][0]) > abs(t[1][0])):
                    self.t = self.t + absolute_scale*self.R.dot(t)
                    self.R = R.dot(self.R)

            # Save the total number of good features
            self.n_features = self.good_new.shape[0]

    def get_mono_coordinates(self):
        # We multiply by the diagonal matrix to fix our vector
        # onto same coordinate axis as true values
        diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])
        adj_coord = np.matmul(diag, self.t)

        return adj_coord.flatten()


    def get_true_coordinates(self):
        '''Returns true coordinates of vehicle
        
        Returns:
            np.array -- Array in format [x, y, z]
        '''
        return self.true_coord.flatten()


    def get_absolute_scale(self):
        '''Used to provide scale estimation for mutliplying
           translation vectors
        
        Returns:
            float -- Scalar value allowing for scale estimation
        '''
        pose = self.pose[self.id - 1].strip().split()
        x_prev = float(pose[3])
        y_prev = float(pose[7])
        z_prev = float(pose[11])
        pose = self.pose[self.id].strip().split()
        x = float(pose[3])
        y = float(pose[7])
        z = float(pose[11])

        true_vect = np.array([[x], [y], [z]])
        self.true_coord = true_vect
        prev_vect = np.array([[x_prev], [y_prev], [z_prev]])
        
        return np.linalg.norm(true_vect - prev_vect)


    def process_frame(self):
        '''Processes images in sequence frame by frame
        '''

        if self.id < 2:
            self.old_frame = cv2.imread(self.file_path +str().zfill(6)+'.png', 0)
            self.current_frame = cv2.imread(self.file_path + str(1).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id = 2
        else:
            self.old_frame = self.current_frame
            self.current_frame = cv2.imread(self.file_path + str(self.id).zfill(6)+'.png', 0)
            self.visual_odometery()
            self.id += 1


