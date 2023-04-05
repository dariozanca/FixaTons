import numpy as np

class RQA(object):

    def __init__(self, s1, s2, r=60, ll=2):
        '''Parameters:
        s1                    (n,2) matrix of x,y fixations
        s2                    (n,2) matrix of x,y fixations
        r                     e.g. 34, radius threshold
        ll                    e.g. 2, minimum line length
        Results:    
        c                     Recurrence matrix
        rec                   Cross-recurrence
        det                   Determinism
        lam                   Laminarity
        corm                  Center of recurrence mass'''

        if s1.shape[1] > 2:
            self.s1 = s1[:,:-1]
            self.s2 = s2[:,:-1]
        else:
            self.s1 = s1
            self.s2 = s2

        self.r = r
        self.ll = ll

        self.l = np.min([len(s1), len(s2)])
        self.c = np.zeros([self.l,self.l])

        self.s1 = s1[:self.l, 0:2]
        self.s2 = s2[:self.l, 0:2]

        for i in range(self.l):
            for j in range(self.l):

                if np.linalg.norm(self.s1[i,:]-self.s2[j,:]) <= self.r:
                    self.c[i,j] = 1

        self.C = np.sum(self.c) + np.finfo(float).eps

    def crossrec(self):
        #C = np.sum(self.c)
        N = self.l
        rec = 100*(self.C/N**2)
        return rec

    def determinism(self):
        #C = np.sum(self.c)
        N = self.l
        D = 0

        for i in range(-N,N):
            d = np.sum(np.diag(self.c, k=i))
            if d >= self.ll:
                D = D + d

        det = 100 * (D/self.C)

        return det

    def laminarity(self):
        #C = np.sum(self.c)
        N = self.l
        H = 0
        V = 0

        for i in range(N):
            h = np.sum(self.c[i,:])
            v = np.sum(self.c[:,i])
            if h >= self.ll:
                H = H + h
            if v >= self.ll:
                V = V + v

        lam = 100 * ((H+V) / (2 * self.C))

        return lam

    def centerrecmass(self):
        #C = np.sum(self.c)
        N = self.l
        Cc = 0
        for i in range(N):
            for j in range(N):
                Cc = Cc + ((j-i) * self.c[i,j])

        corm = 100*(Cc/((N-1)*self.C))

        return corm

    def compute_rqa_metrics(self):
        determinism = self.determinism()
        laminarity = self.laminarity()
        corm = self.centerrecmass()
        crossrec = self.crossrec()

        scores = np.array([determinism, corm, crossrec, laminarity])

        return 