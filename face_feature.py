import math

class Feature:
    points_51 = [] # 51 key points model (different order)
    points_68 = [] # original 68 key points model
    feature = [None] * 29 # f0-f28
    A = 0
    B_L = B_R = 0
    C = 0
    D = E = 0
    F = G = 0
    H = I = 0
    J = K = 0
    
    L = M = 0
    N_L = N_R = O_L = O_R = 0
    N = 0
    O = 0
    P_L = P_U = Q_L = Q_U = 0

    R = S = 0
    T = U = 0
    V_L = V_R = 0
    W = 0
    X = 0

    def __init__(self, points_68):
        if len(points_68) != 68:
            print("Not 68 landmarks")
            exit()

        self.points_68 = points_68
        self.points_51 = Feature.points_68_to_51(points_68)

        self.A = self.dist(self.points_51[48], self.points_51[49])
        self.B_L = self.dist(self.points_51[10], self.points_51[13])
        self.B_R = self.dist(self.points_51[16], self.points_51[19])

        self.C = self.dist(self.points_51[37], self.points_51[50])

        self.D = self.dist(self.points_51[48], self.points_51[10])
        self.E = self.dist(self.points_51[19], self.points_51[49])

        self.F = self.dist(self.points_51[10], self.points_51[37])
        self.G = self.dist(self.points_51[19], self.points_51[37])
        
        self.H = self.dist(self.points_51[10], self.points_51[23])
        self.I = self.dist(self.points_51[19], self.points_51[27])

        self.J = self.dist(self.points_51[23], self.points_51[37])
        self.K = self.dist(self.points_51[27], self.points_51[37])

        self.L = sum([p.y for p in self.points_51[0:4 + 1]]) / 5
        self.M = sum([p.y for p in self.points_51[5:9 + 1]]) / 5

        self.N_L = self.dist(self.points_51[11], self.points_51[15])
        self.N_R = self.dist(self.points_51[12], self.points_51[14])
        self.O_L = self.dist(self.points_51[17], self.points_51[21])
        self.O_R = self.dist(self.points_51[18], self.points_51[20])

        self.N = (self.N_L + self.N_R) / 2
        self.O = (self.O_L + self.O_R) / 2

        self.P_L = self.dist(self.points_51[29], self.points_51[39])
        self.P_U = self.dist(self.points_51[30], self.points_51[38])
        self.Q_L = self.dist(self.points_51[33], self.points_51[35])
        self.Q_U = self.dist(self.points_51[32], self.points_51[36])

        self.R = self.dist(self.points_51[3], self.points_51[37])
        self.S = self.dist(self.points_51[6], self.points_51[37])

        self.T = self.dist(self.points_51[2], self.points_51[37])
        self.U = self.dist(self.points_51[7], self.points_51[37])

        self.V_L = self.dist(self.points_51[28], self.points_51[37])
        self.V_R = self.dist(self.points_51[34], self.points_51[37])

        self.W = self.dist(self.points_51[28], self.points_51[34])
        
        self.X = self.dist(self.points_51[22], self.points_51[31])
        
        # Eyebrows
        self.feature[0] = abs(self.angle(self.points_51[0], self.points_51[9]))
        self.feature[1] = abs(self.angle(self.points_51[2], self.points_51[7])) 
        self.feature[2] = abs(self.angle(self.points_51[4], self.points_51[5])) 
        self.feature[3] = max(self.L / self.M, self.M / self.L)
        self.feature[4] = self.slope(self.points_51[0], self.points_51[9])
        self.feature[5] = self.slope(self.points_51[2], self.points_51[7])
        self.feature[6] = self.slope(self.points_51[4], self.points_51[5])

        # Eyes
        self.feature[7] = abs(self.angle(self.points_51[10], self.points_51[19]))
        self.feature[8] = max(self.B_L / self.B_R, self.B_R / self.B_L)
        self.feature[9] = max(self.D / self.E, self.E / self.D)
        self.feature[10] = max(self.H / self.I, self.I / self.H)
        self.feature[11] = max(self.N / self.O, self.O / self.N)
        self.feature[12] = max(self.N_L / self.O_R, self.O_R / self.N_L)
        self.feature[13] = max(self.N_R / self.O_L, self.O_L / self.N_R)
        
        # Mouth
        self.feature[14] = abs(self.angle(self.points_51[28], self.points_51[34]))
        self.feature[15] = max(self.F / self.G, self.G / self.F)
        self.feature[16] = max(self.P_L / self.Q_L, self.Q_L / self.P_L)
        self.feature[17] = max(self.P_U / self.Q_U, self.Q_U / self.P_U)
        self.feature[18] = max(self.V_L / self.A, self.V_R / self.A)
        self.feature[19] = max(self.P_L / self.W, self.Q_L / self.W)
        self.feature[20] = max(self.P_U / self.W, self.Q_U / self.W)

        left_lips = self.points_51[29:31 + 1] + self.points_51[37:39 + 1]
        right_lips = self.points_51[29:37 + 1]
        self.feature[21] = max(self.segment(left_lips) / self.W, self.segment(right_lips) / self.W)

        # Nose
        self.feature[22] = abs(self.angle(self.points_51[23], self.points_51[27]))

        # Combined
        self.feature[23] = abs(self.angle(self.points_51[22], self.points_51[37]))
        self.feature[24] = max(self.J / self.K, self.K / self.J)
        self.feature[25] = max(self.T/ self.A, self.U / self.A)
        self.feature[26] = max(self.R / self.A, self.S / self.A)
        self.feature[27] = self.C / self.A
        self.feature[28] = self.X / self.A

    # Given point a and b return their euclidean distance
    @staticmethod
    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    
    # Given point a and b return their slope
    @staticmethod
    def slope(a, b):
        return abs((a.y - b.y) / (a.x - b.x))
    
    # Given point a and b return angle of delta x and y in degree
    @staticmethod
    def angle(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        return math.atan2(dy, dx) * 180 / math.pi
    
    # Given point in order, 
    @staticmethod
    def segment(points):
        sum = 0
        for i in range(len(points) - 1):
            sum += Feature.dist(points[i], points[i + 1])
        sum += Feature.dist(points[0], points[-1])
        return sum
        

    @staticmethod
    def points_68_to_51(points_68):
        points_51 = [None] * 51

        points_51[0:4 + 1] = points_68[17:21 + 1]
        points_51[5:9 + 1] = points_68[22:26 + 1]
        points_51[10:15 + 1] = points_68[36:41 + 1]
        points_51[16:21 + 1] = points_68[42:47 + 1]
        points_51[22:27 + 1] = points_68[30:35 + 1]
        points_51[28:47 + 1] = points_68[48:67 + 1]
        points_51[48] = points_68[0]
        points_51[49] = points_68[16]
        points_51[50] = points_68[8]
        return points_51
    