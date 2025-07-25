import numpy as np
import cv2

class UrCustom():
    def __init__(self, robot_type="ur5e"):
        """
        URロボットのカスタムクラス
        
        Parameters
        ----------
        robot_type : str
            ロボットの種類
        
        Attributes
        ----------
        _a : np.array
            リンクの長さ
        _d : np.array
            リンクのオフセット
        """                
                
        if robot_type == "ur5e":
            self._a = np.array([0, -0.4250, -0.3922, 0, 0, 0])
            self._d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        elif robot_type == "ur10e":
            self._a = np.array([0, -0.6127, -0.57155, 0, 0, 0])
            self._d = np.array([0.1807, 0, 0, 0.17415, 0.11985, 0.11655])
        elif robot_type == "fr5":
            self._a = np.array([0, -0.425, -0.395, 0, 0, 0])
            self._d = np.array([0.152, 0, 0, 0.102, 0.102, 0.100])
        else:
            print("robot type is not defined")
            exit(-1)
        
        #Initialize
        self._T01 = np.array([])
        self._T02 = np.array([])
        self._T03 = np.array([])
        self._T04 = np.array([]) 
        self._T05 = np.array([])
        self._T06 = np.array([])
        #link transformation matrix
        self._T12 = np.array([])
        self._T23 = np.array([])
        self._T34 = np.array([]) 
        self._T45 = np.array([])
        self._T56 = np.array([])

        #jacobian
        self.J = np.array([])

    def _mat01(self, theta1):
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        
        self._T01 = np.array([
            [c1, 0.0, s1, 0.0],
            [s1, 0.0, -c1, 0.0],
            [0.0, 1.0, 0.0, self._d[0]],
            [0.0, 0.0, 0.0, 1.0]
        ])

    def _mat12(self, theta2):
        c2 = np.cos(theta2)
        s2 = np.sin(theta2)
        
        self._T12 = np.array([
            [c2, -s2, 0.0, self._a[1] * c2],
            [s2, c2, 0.0, self._a[1] * s2],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
    def _mat23(self, theta3):
        c3 = np.cos(theta3)
        s3 = np.sin(theta3)
        
        self._T23 = np.array([
            [c3, -s3, 0.0, self._a[2] * c3],
            [s3, c3, 0.0, self._a[2] * s3],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])  
        
    def _mat34(self, theta4):
        c4 = np.cos(theta4)
        s4 = np.sin(theta4)
        
        self._T34 = np.array([
            [c4, 0.0, s4, 0.0],
            [s4, 0.0, -c4, 0.0],
            [0.0, 1.0, 0.0, self._d[3]],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
    def _mat45(self, theta5):
        c5 = np.cos(theta5)
        s5 = np.sin(theta5)
        
        self._T45 = np.array([
            [c5, 0.0, -s5, 0.0],
            [s5, 0.0, c5, 0.0],
            [0.0, -1.0, 0.0, self._d[4]],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
    def _mat56(self, theta6):
        c6 = np.cos(theta6)
        s6 = np.sin(theta6)
        
        self._T56 = np.array([
            [c6, -s6, 0.0, 0.0],
            [s6, c6, 0.0, 0.0],
            [0.0, 0.0, 1.0, self._d[5]],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
    def mat(self, joints):
        self._mat01(joints[0])
        self._mat12(joints[1])
        self._mat23(joints[2])
        self._mat34(joints[3])
        self._mat45(joints[4])
        self._mat56(joints[5])
        
    def _cal_pose1(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)

        n1, _ = cv2.Rodrigues(self._T01[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.
        
        angle = self._norm_rotVec(n1)

        self.pose1 = np.array([self._T01[0, 3], self._T01[1, 3], self._T01[2, 3], 
                               n1[0][0]/angle, n1[1][0]/angle, n1[2][0]/angle]) #(x,y,z,rx,ry,rz)
        
    def _cal_pose2(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T02 = self._T01 @ self._T12
        n2, _ = cv2.Rodrigues(self._T02[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.

        angle = self._norm_rotVec(n2)

        self.pose2 = np.array([self._T02[0, 3], self._T02[1, 3], self._T02[2, 3], 
                               n2[0][0]/angle, n2[1][0]/angle, n2[2][0]/angle])
        
    def _cal_pose3(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T03 = self._T01 @ self._T12 @ self._T23
        n3, _ = cv2.Rodrigues(self._T03[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.
        
        angle = self._norm_rotVec(n3)

        self.pose3 = np.array([self._T03[0, 3], self._T03[1, 3], self._T03[2, 3], 
                               n3[0][0]/angle, n3[1][0]/angle, n3[2][0]/angle])
        
    def _cal_pose4(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T04 = self._T01 @ self._T12 @ self._T23 @ self._T34
        n4, _ = cv2.Rodrigues(self._T04[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.
        
        angle = self._norm_rotVec(n4)

        self.pose4 = np.array([self._T04[0, 3], self._T04[1, 3], self._T04[2, 3], 
                               n4[0][0]/angle, n4[1][0]/angle, n4[2][0]/angle])
        
    def _cal_pose5(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T05 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45
        n5, _ = cv2.Rodrigues(self._T05[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.
        
        angle = self._norm_rotVec(n5)
        self.pose5 = np.array([self._T05[0, 3], self._T05[1, 3], self._T05[2, 3], n5[0][0]/angle, n5[1][0]/angle, n5[2][0]/angle])
        
    def _cal_pose6(self, joints, bool_mat):
        if bool_mat:
            self.mat(joints)
        
        self._T06 = self._T01 @ self._T12 @ self._T23 @ self._T34 @ self._T45 @ self._T56
        n6, _ = cv2.Rodrigues(self._T06[:3,:3]) #convert rotation matrix to roattion vector with Rodrigues formula.
        
        angle = self._norm_rotVec(n6)

        self.pose6 = np.array([self._T06[0, 3], self._T06[1, 3], self._T06[2, 3], n6[0][0]/angle, n6[1][0]/angle, n6[2][0]/angle])
    
    def _norm_rotVec(self,n):
        return (n[0][0]**2+n[1][0]**2+n[2][0]**2+1e-16)**(0.5)
    
    def cal_pose_all(self, joints):
        self.mat(joints)
        
        self._cal_pose1(joints, False)
        self._cal_pose2(joints, False)
        self._cal_pose3(joints, False)
        self._cal_pose4(joints, False)
        self._cal_pose5(joints, False)
        self._cal_pose6(joints, False)
    
    def cal_jacobian01(self, joints):
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]
        
        self.J10 = self._jacobian(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 1)
        
    def cal_jacobian02(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]
        
        self.J02 = self._jacobian(s1, c1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, px, py, pz, 2)
    
    def cal_jacobian03(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J03 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 3)
        
    def cal_jacobian04(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self._T06[0, 3]
        py = self._T06[1, 3]
        pz = self._T06[2, 3]
        
        self.J04 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 4)
            
    def cal_jacobian05(self, joints):
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        px = self.pose6[0]
        py = self.pose6[1]
        pz = self.pose6[2]

        self.J05 = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, 0.0, 0.0, 0.0, px, py, pz, 5)
        
    def cal_jacobian(self, joints):
        """
        ヤコビアンを求める
        
        Parameters
        ----------
        joints : np.array
            関節角度
        
        Attributes
        ----------
        J : np.array
            ヤコビアン
        """
        
        s1 = np.sin(joints[0])
        c1 = np.cos(joints[0])
        s23 = np.sin(joints[1] + joints[2])
        c23 = np.cos(joints[1] + joints[2])
        s234 = np.sin(joints[1] + joints[2] + joints[3])
        c234 = np.cos(joints[1] + joints[2] + joints[3])
        s5 = np.sin(joints[4])
        c5 = np.cos(joints[4])
        r13 = self._T06[0, 2]
        r23 = self._T06[1, 2]
        r33 = self._T06[2, 2]
        px = self._T06[0, 3]
        py = self._T06[1, 3]
        pz = self._T06[2, 3]
        
        self.J = self._jacobian(s1, c1, s23, c23, s234, c234, s5, c5, r13, r23, r33, px, py, pz, 6)
        
    def _jacobian(self, s1, c1, s23, c23, s234, c234, s5, c5, r13, r23, r33, px, py, pz, index):
        j00 = -py
        j10 = px
        j20 = 0.0
        j30 = 0.0
        j40 = 0.0
        j50 = 1.0
        j01 = -c1 * (pz - self._d[0])
        j11 = -s1 * (pz - self._d[0])
        j21 = s1 * py + c1 * px
        j31 = s1
        j41 = -c1
        j51 = 0.0
        j02 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j12 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4] - s23 * self._a[2])
        j22 = -c234 * s5 * self._d[5] + s234 * self._d[4] + c23 * self._a[2]
        j32 = s1
        j42 = -c1
        j52 = 0.0
        j03 = c1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j13 = s1 * (s234 * s5 * self._d[5] + c234 * self._d[4])
        j23 = -c234 * s5 * self._d[5] + s234 * self._d[4]
        j33 = s1
        j43 = -c1
        j53 = 0.0
        j04 = -self._d[5] * (s1 * s5 + c1 * c234 * c5)
        j14 = self._d[5] * (c1 * s5 - s1 * c234 * c5)
        j24 = -c5 * s234 * self._d[5]
        j34 = c1 * s234
        j44 = s1 * s234
        j54 = -c234
        j05 = 0.0
        j15 = 0.0
        j25 = 0.0
        j35 = r13
        j45 = r23
        j55 = r33
        
        j = np.array([
            [j00, j01, j02, j03, j04, j05],
            [j10, j11, j12, j13, j14, j15],
            [j20, j21, j22, j23, j24, j25],
            [j30, j31, j32, j33, j34, j35],
            [j40, j41, j42, j43, j44, j45],
            [j50, j51, j52, j53, j54, j55]
        ])
        
        return j[:, :index]
    
    def determinant(self, j):
        """
        ヤコビアンの行列式を求める
        
        Parameters
        ----------
        j : np.array
            ヤコビアン
        
        Returns
        -------
        det : float
            行列式
        """
        
        if j.shape[0] == j.shape[1]:
            return np.linalg.det(j)

        jT = np.transpose(j)
        jTj = jT @ j
        
        return np.sqrt(np.linalg.det(jTj))