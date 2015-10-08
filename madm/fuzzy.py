# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 07:47:56 2013

@author: - Frank Patterson

--- fuzzy.py ---

handles operations on fuzzy sets
    
Version 1.0 (15Dec13) - 
    First release version. Fuzzy Set Data Classes not used yet. Fuzzy Operators
    for fuzzy triangular are complete, but not fuzzy trapezoidal
    
"""

#------------------------------------------------------------------------
#FUZZY SET DATA CLASSES
#------------------------------------------------------------------------

    
#linear fuzzy set

class linearSet:
    # define the linear set as a fuzzy set with points [x,m], where 
    # 0 < m < 1 and [x,m] defines the membership function of the set
    # m(x) < x(min) or m(x) > x(max) are assumed to be = 0
    
    def __init__(self, *args):
        # input args define [x,m] verticies of set
        self.type = 'linear'
        self.vertices = []
        try:
            for count, arg in enumerate(*args):
                if len(arg) == 2:
                    if 0.0 <= arg[1] <= 1.0:
                        self.vertices.append([float(arg[0]), float(arg[1])])
            self.vertices.sort(key=lambda x: x[0])
        except: None
        

#------------------------------------------------------------------------
#FUZZY OPERATORS
#------------------------------------------------------------------------


def add_FuzzyTri(*args):
    #takes in any number of triangular fuzzy numbers: mu(x1,x2,x3) = (0,1,0)
    #all args in form of list of numbers [a1, a2, a3] where a1 < a2 < a3 
    #returns the sum [b1 = sum(a1), b2 = sum(a2), b3 = sum(a3)]
    try:
        fzysum = [0,0,0]
        for count, arg in enumerate(*args):
            fzysum[0] = fzysum[0] + arg[0]
            fzysum[1] = fzysum[1] + arg[1]
            fzysum[2] = fzysum[2] + arg[2]
        return fzysum
    except:
        return None
    
    
def mult_FuzzyTri(*args):
    #takes in any number of triangular fuzzy numbers: mu(x1,x2,x3) = (0,1,0)
    #all args in form of list of numbers [a1, a2, a3] where a1 < a2 < a3 
    #returns the product [b1 = product(a1), b2 = product(a2), b3 = product(a3)]
    #try:
    fzyprd = [1,1,1]
    for count, arg in enumerate(*args):
        fzyprd[0] = fzyprd[0] * arg[0]
        fzyprd[1] = fzyprd[1] * arg[1]
        fzyprd[2] = fzyprd[2] * arg[2]
    return fzyprd
    #except:
    #    return None
    
def divide_FuzzyTri(a,b):
    #takes 2 triangular fuzzy numbers: mu(x1,x2,x3) = (0,1,0)
    #all args in form of list of numbers [a1, a2, a3] where a1 < a2 < a3 
    #returns the result of  a/b
    #try:
    fzy_result = [min(a)/max(b), a[1]/b[1], max(a)/min(b)]
    return fzy_result
    #except:
    #    return None
        
        
def dominance_FuzzyTri(A,B):
    #takes in two triangular fuzzy numbers and returns the value for the dominance d(A,B)
    #as defined by Tong & Bonnisone (1980) d(A,B) = max( min( mu_lesA(x), mu_B(x) ) )
    #returns the dominance [0, 1.0]
    try:
        if A[1] > B[1]: return 1.0
        elif A[2] < B[0]: return 0.0
        else:    
            return -(A[2] - B[0])/(-A[2] + A[1] - B[1] + B[0])
    except:
        return None


def getAlphaCutTri(a,b,c,alpha):
    #takes in a triangular fuzzy number and alpha value ([0,1]) and returns the
    #alpha cut in the form x,y
    x = (b-a)*alpha + a
    y = c - (c-b)*alpha
    return x,y

def add_FuzzyTrap(*args):
    #takes in any number of triangular fuzzy numbers: mu(x1,x2,x3) = (0,1,0)
    #all args in form of list of numbers [a1, a2, a3] where a1 < a2 < a3 
    #returns the sum [b1 = sum(a1), b2 = sum(a2), b3 = sum(a3)]
    #try:
    fzysum = [0,0,0,0]
    for count, arg in enumerate(*args):
        fzysum[0] = fzysum[0] + arg[0]
        fzysum[1] = fzysum[1] + arg[1]
        fzysum[2] = fzysum[2] + arg[2]
        fzysum[3] = fzysum[3] + arg[3]
    return fzysum
    #except:
    #    return None
    
    
def mult_FuzzyTrap(*args):
    #takes in any number of triangular fuzzy numbers: mu(x1,x2,x3) = (0,1,0)
    #all args in form of list of numbers [a1, a2, a3] where a1 < a2 < a3 
    #returns the product [b1 = product(a1), b2 = product(a2), b3 = product(a3)]
    #try:
    fzyprd = [1,1,1,1]
    for count, arg in enumerate(*args):
        fzyprd[0] = fzyprd[0] * arg[0]
        fzyprd[1] = fzyprd[1] * arg[1]
        fzyprd[2] = fzyprd[2] * arg[2]
        fzyprd[3] = fzyprd[3] * arg[3]
    return fzyprd
    #except:
    #    return None
    
def divide_FuzzyTrap(a,b):
    #takes 2 trapezoidal fuzzy numbers: mu(x1,x2,x3,x4) = (0,1,1,0)
    #all args in form of list of numbers [a1, a2, a3, a4] where a1 <= a2 <= a3 <= a4
    #returns the result of  a/b
    #try:
    fzy_result = [a[0]/b[3],
                  a[1]/b[2],
                  a[2]/b[1],
                  a[3]/b[0]]
    return fzy_result
    #except:
    #    return None
                
        
def dominance_FuzzyTrap(A,B):
    #takes in two triangular fuzzy numbers and returns the value for the dominance d(A,B)
    #essentially the possibility that A > B
    #as defined by Tong & Bonnisone (1980) d(A,B) = max( min( mu_lesA(x), mu_B(x) ) )
    #returns the dominance [0, 1.0]
    try:
        if   A[1] > B[2]:                 return 1.0
        elif A[2] < B[2] and A[2] > B[1]: return 1.0
        elif B[2] < A[2] and B[2] > A[1]: return 1.0
        elif A[3] < B[0]:                 return 0.0
        else:    
            return -(A[3] - B[0])/(-A[3] + A[2] - B[1] + B[0])
    except:
        return None
        
def getAlphaCutTrap(a,b,c,d, alpha):
    #takes in a triangular fuzzy number and alpha value ([0,1]) and returns the
    #alpha cut in the form x,y
    x = (b-a)*alpha + a
    y = d - (d-c)*alpha
    return x,y


def getAlphaCut(fuzzyNum,alpha):
    #takes in a fuzzy number and alpha value ([0,1]) and returns the
    #fuzzy TRI: 
    if len(fuzzyNum) == 3 or len(fuzzyNum) == 4:
        x = (fuzzyNum[1]-fuzzyNum[0])*alpha + fuzzyNum[0]
        y = fuzzyNum[-1] - (fuzzyNum[-1]-fuzzyNum[-2])*alpha
        return x,y
        
        
def dominance_AlphaCut(A_alphas, A_cut_intervals, B_alphas, B_cut_intervals):
    #takes in alpha levels and the resultng cuts for two fuzzy numbers (A, B)
    #finds the resulting dominance 
    #essentially the possibility that A > B
    #as defined by Tong & Bonnisone (1980) d(A,B) = max( min( mu_lesA(x), mu_B(x) ) )
    #returns the dominance [0, 1.0]
    
    #sort cuts into list of fuzzy function
    #Ax1 = [a[0] for a in A_cut_intervals]
    Ax2 = [A_cut_intervals[len(A_alphas)-1-i][1] for i in range(len(A_alphas))]
    #Ay1 = A_alphas
    Ay2 = [A_alphas[len(A_alphas)-1-i] for i in range(len(A_alphas))]
    A_L = [Ax2,Ay2]         #right half of fuzzy function A [x,y]

    Bx1 = [b[0] for b in B_cut_intervals]
    #Bx2 = [B_cut_intervals[len(B_alphas)-1-i][1] for i in range(len(B_alphas))]
    By1 = B_alphas
    #By2 = [B_alphas[len(B_alphas)-1-i] for i in range(len(B_alphas))]
    B_R = [Bx1,By1]         #left half of fuzzy function B [x,y]    
    
    if min(A_L[0]) >= max(B_R[0]): 
        #print 'case1'        
        return 1.0 
    elif  max(A_L[0]) <= min(B_R[0]): 
        #print 'case2', 'Max A =', max(A_L[0]), 'Min B =' , min(B_R[0]) 
        return 0.0
    else: 
        #print 'case3'  
        for i in range(1, len(A_L[0])):
            for j in range(1, len(B_R[0])):
                if B_R[0][j-1] <= A_L[0][i-1] <= B_R[0][j] and \
                   A_L[0][i] > B_R[0][j] or \
                   B_R[0][j-1] <= A_L[0][i] <= B_R[0][j] and \
                   A_L[0][i-1] < B_R[0][j-1]:   #if line segments intersect
                       x1 = A_L[0][i-1]
                       x2 = A_L[0][i]
                       x3 = B_R[0][j-1]
                       x4 = B_R[0][j]
                       y1 = A_L[1][i-1]
                       y2 = A_L[1][i]
                       y3 = B_R[1][j-1]
                       y4 = B_R[1][j]
                       #I_x = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/ \
                       #      ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
                       I_y = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/ \
                             ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
                       return I_y
                   
    

    
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#-----------------------------TESTING------------------------------------
#------------------------------------------------------------------------
if __name__=="__main__": 
    
    import random
    import matplotlib.pyplot as plt



    """    
    "Test Linear Set... " 
    for i in range(25):
        pts = ()
        pts = [[random.randint(1,10),float(str(random.uniform(0,1))[0:5])] for i in range(random.randint(3,7))]
        pts = tuple(pts)
        l = linearSet(pts)
        print "Test Points: ", pts
        print "Vertices: ", l.vertices, '\n'

    print "\n", "Test add_FuzzyTri & mult_FuzzyTri..."
    
    for i in range(10):
        pts = ()
        pts = [[float(str(random.uniform(1,9))[0:5]), \
                       float(str(random.uniform(1,9))[0:5]), \
                       float(str(random.uniform(1,9))[0:5])] for i in range(random.randint(3,7))]
        pts = tuple(pts)
        for p in pts: 
            p.sort()
        print 'FTNs:', pts
        print 'SUM:  ', add_FuzzyTri(pts)
        print 'PRODUCT:  ', mult_FuzzyTri(pts), '\n'
        
        
    print "\n", "Test Dominance"
    
    fig = plt.figure()
    for i in range(4):
        A = [float(str(random.uniform(1,9))[0:5]), float(str(random.uniform(0,9))[0:5]), \
                       float(str(random.uniform(1,9))[0:5])]
        B = [float(str(random.uniform(1,9))[0:5]), float(str(random.uniform(0,9))[0:5]), \
                       float(str(random.uniform(0,9))[0:5])]
        A.sort()
        B.sort()
        plt.subplot((221+i))
        plt.plot(A, [0.0, 1.0, 0.0], linewidth=2.0)
        plt.plot(B, [0.0, 1.0, 0.0], linewidth=2.0)
        plt.plot([0,9], [dominance_FuzzyTri(A,B),dominance_FuzzyTri(A,B)], '--c')
        plt.axes = [1, 9, 0., 1.0]
        plt.legend(['A','B'])
        plt.title(str(i+1))
        print "Case", (i+1), ":   A=", A, "      B=", B, "   Dom:", dominance_FuzzyTri(A,B)
    plt.show()

    fig = plt.figure()
    for i in range(4):
        A = [float(str(random.uniform(1,9))[0:5]), float(str(random.uniform(0,9))[0:5]), \
                       float(str(random.uniform(1,9))[0:5]), float(str(random.uniform(0,9))[0:5])]
        B = [float(str(random.uniform(1,9))[0:5]), float(str(random.uniform(0,9))[0:5]), \
                       float(str(random.uniform(0,9))[0:5]), float(str(random.uniform(0,9))[0:5])]
        A.sort()
        B.sort()
        plt.subplot((221+i))
        plt.plot(A, [0.0, 1.0, 1.0, 0.0], linewidth=2.0)
        plt.plot(B, [0.0, 1.0, 1.0, 0.0], linewidth=2.0)
        plt.plot([0,9], [dominance_FuzzyTrap(A,B),dominance_FuzzyTrap(A,B)], '--c')
        plt.axes = [1, 9, 0., 1.05]
        plt.legend(['A','B'])
        plt.title(str(i+1))
        print "Case", (i+1), ":   A=", A, "      B=", B, "   Dom:", dominance_FuzzyTrap(A,B)
    plt.show()
    """
    
    #test alpha cut dominance
    RC5 = [[0.5108206329848081, 0.9741568648767817], [0.5211260734228694, 0.9659158650627858], [0.5327478632075343, 0.9571294582225801], [0.5455962636898543, 0.9479851820632369], [0.5595594093523893, 0.9385270278205919], [0.574462102581285, 0.9288487490866657], [0.5900920770460122, 0.9190391630142851], [0.6062736167776475, 0.9091066196298371], [0.6227833750703438, 0.8991437836195492], [0.6394232694474807, 0.8902538861604626], [0.655988352207335, 0.8833080768065414], [0.6723718845180665, 0.8760406146476102], [0.6884198137530917, 0.8684574934061262], [0.7040005036945519, 0.8605705554444283], [0.719083339072118, 0.8523429807865381], [0.7335986925062086, 0.8437911091246695], [0.7475023303324492, 0.8349221843163305], [0.7607482870905863, 0.8257705452352337], [0.7733838281004883, 0.8166019592817633], [0.7853871762606269, 0.80693641706244], [0.796726360412427, 0.796726360412427]]
    RC4 = [[0.184983400548751, 0.9660994911633647], [0.21334285849527113, 0.9561183482342258], [0.23935256959512116, 0.9454767284114062], [0.261193736788885, 0.9342881544953004], [0.2842321066625626, 0.9224656012884602], [0.308213320653759, 0.9101407728015585], [0.33289388063352576, 0.8973446119925647], [0.35817251546585843, 0.8840012009174532], [0.38386002764575083, 0.8701523216397544], [0.40978596568557707, 0.8558659873302927], [0.4358008818200046, 0.841191090353683], [0.46183086945114254, 0.8260659697656445], [0.48775667949194274, 0.8105399489423057], [0.5134022351734842, 0.7956280796854817], [0.5387542767221954, 0.7810615115831874], [0.5636880734647467, 0.7658351522367086], [0.5873148706105751, 0.749904825699825], [0.6105177924511899, 0.7333261292518766], [0.6336737670854127, 0.7159868706752659], [0.6566354586053879, 0.6979097547685303], [0.6791906544662232, 0.6791906544662232]]
    RC3 = [[0.4974699381298749, 0.9687786879640573], [0.5082670590037185, 0.9596134987715004], [0.5201814020251726, 0.9493935974050522], [0.533095170097744, 0.938448308812271], [0.5468536751076027, 0.9269433787022829], [0.5612338754676113, 0.9150226140842824], [0.5760208908554967, 0.9027609472373469], [0.5909854627778599, 0.8902400512164519], [0.6059266247466466, 0.8775487345300353], [0.6206144440803103, 0.8647544091021118], [0.6348860499484285, 0.8520840968943978], [0.6486134141037495, 0.8420399418565608], [0.6617176018968758, 0.8316444130940726], [0.6703377132901872, 0.8209372710540497], [0.6790444134507011, 0.8099377550705099], [0.6882227343712596, 0.7986823990227064], [0.6978768681689451, 0.7872062750185815], [0.7079669854536312, 0.7755867589236547], [0.7184750411856656, 0.7638794987398623], [0.7293604962084269, 0.7521604412961288], [0.740542837882752, 0.740542837882752]]
    RC2 = [[0.4584830964498254, 0.8716798547462277], [0.4594185556116653, 0.8709397721468237], [0.4604850720711951, 0.8700961697451146], [0.461691301208789, 0.8691408826934742], [0.46304594080727646, 0.8680648346225736], [0.4645598620225738, 0.8668556893059605], [0.4662376237962298, 0.8655189780870048], [0.46808947885628716, 0.8640402067959363], [0.4701235581224106, 0.8624121059692048], [0.47234763609072794, 0.860627699416686], [0.4747735556486775, 0.8586663954530491], [0.47739965221835257, 0.8565491793991197], [0.48023603378338936, 0.8542544572415696], [0.48328833881273897, 0.8517824086569747], [0.4865608224019448, 0.849131996510137], [0.49006493031150944, 0.8462674216786993], [0.493788512930994, 0.8432302818663061], [0.49738883441149295, 0.8399988682493097], [0.5009063202438504, 0.836571242819525], [0.5046406752923551, 0.8329492200513394], [0.5086070515861575, 0.8291019674913392]]    
    RC1 = [[0.30085363637692236, 0.9627060326661859], [0.31016889067430675, 0.9583397545614085], [0.3197573390602781, 0.9538460054707077], [0.32960234147817946, 0.9492469518956578], [0.33969959300086366, 0.9444622797374349], [0.35000726220701317, 0.9395931665153807], [0.36051885382514615, 0.9345994959448254], [0.37121613219726424, 0.92948109440095], [0.38208062537691934, 0.9242571053222416], [0.3930936767903836, 0.9188858863747814], [0.4042364927921704, 0.9133929404854408], [0.4154901871350953, 0.9077762745275821], [0.4268604176085658, 0.9019979134033173], [0.4382804589181629, 0.8961555532900265], [0.4497545395624668, 0.8901722072245303], [0.461263805504465, 0.8840769556719245], [0.4727895115828503, 0.8778693102639215], [0.48431306066258034, 0.8715534631327689], [0.4958160429523125, 0.865149629056961], [0.5072802755621427, 0.8586269266297498], [0.5186878423258366, 0.8520183709318127]]    
    levels = 21
    alphas = [float(n)/(levels-1) for n in range(levels)]


    RC_A = RC4
    RC_B = RC3
    
    d = dominance_AlphaCut(alphas, RC_A, alphas, RC_B)
    print d
    
    #sort cuts into list of fuzzy function
    Ax1 = [a[0] for a in RC_A]
    Ax2 = [RC_A[len(alphas)-1-i][1] for i in range(len(alphas))]
    Ay1 = alphas
    Ay2 = [alphas[len(alphas)-1-i] for i in range(len(alphas))]
    A_L = [Ax2, Ay2]         # left half of fuzzy function A [x,y]

    Bx1 = [b[0] for b in RC_B]
    Bx2 = [RC_B[len(alphas)-1-i][1] for i in range(len(alphas))]
    By1 = alphas #[1.0 for i in range(len(B_alphas))]
    By2 = [alphas[len(alphas)-1-i] for i in range(len(alphas))]
    B_R = [Bx1, By1]         #right half of fuzzy function B [x,y]  
    
    plt.figure()
    plt.plot(A_L[0],A_L[1], '-r')
    plt.plot(B_R[0],B_R[1], '-b')
    plt.plot([0,1],[d,d], '--k')
    plt.show()