import numpy as np
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def __make_scatter(self):
        plt.scatter(self.x, self.y, label="Data Points")

    def __reg(self):
        x = self.x
        y = self.y

        x2 = x**2
        xy = x*y
        n = len(x)

        a = (sum(y)*sum(x2) - sum(x)*sum(xy))/(n*sum(x2) - sum(x)**2)
        b = (n*sum(xy) - sum(x)*sum(y))/(n*sum(x2) - sum(x)**2)

        x = np.insert(x, 0, 0)

        return x, a, b

    def predict(self, y):
        x, a, b = self.__reg()
        return a + b*y

    def get_scat(self):
        plt.scatter(self.x, self.y, label="Data Points")
        plt.title("Scatter Plot")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.xlim(0, (max(self.x)+1))
        plt.legend()
        plt.show()

    def get_reg(self):
        x, a, b = self.__reg()
        plt.plot(x, a + b*x, linestyle='solid', color="red", label="Line of Best Fit")
        plt.title("Linear Regression")
        plt.xlim(0, (max(x)+1))
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.legend()
        plt.show()

    def get_comp(self):
        self.__make_scatter()
        x, a, b = self.__reg()
        plt.plot(x, a + b*x, linestyle='solid', color="red", label="Line of Best Fit")
        plt.title("Linear Regression & Scatter Plot")
        plt.xlim(0, (max(x)+1))
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.legend()
        plt.show()

    def get_corcof(self):
        x = self.x
        y = self.y

        n = len(x)
        xy = x*y
        x2 = x**2
        y2 = y**2

        r = (n*sum(xy) - sum(x)*sum(y)) / \
            np.sqrt((n*sum(x2) - sum(x)**2)*(n*sum(y2) - sum(y)**2))

        return r

class MulReg():
    def __init__(self, x1, x2, y):
        self.x1 = np.array(x1)
        self.x2 = np.array(x2)
        self.y = np.array(y)

    def get_scat(self): #Scatterplots
        x1 = self.x1
        x2 = self.x2
        y  = self.y
        fig = plt.figure()

        ax = fig.add_subplot(2,2,1, xlabel="X1", ylabel="X2")
        ax.scatter(x1,x2)
        

        ax = fig.add_subplot(2,2,2, xlabel="X1", ylabel="Y")
        ax.scatter(x1,y)

        ax = fig.add_subplot(2,2,3, xlabel="X2", ylabel="Y")
        ax.scatter(x2,y)

        ax = fig.add_subplot(2,2,4, projection="3d", xlabel="X1", ylabel="X2", zlabel="Y")
        ax.scatter(x1,x2,y)

        plt.show()

    def get_coeffs(self): # Returns Coefficients
        x1 = self.x1
        x2 = self.x2
        y  = self.y
        x1_sq = x1**2
        x2_sq = x2**2
        x1y = x1*y
        x2y = x2*y
        x1x2 = x1*x2
        n = len(x1)

        ### Summations
        sum_x1    = np.sum(x1)
        sum_x2    = np.sum(x2)
        sum_y     = np.sum(y)
        sum_x1_sq = np.sum(x1_sq)
        sum_x2_sq = np.sum(x2_sq)
        sum_x1y   = np.sum(x1y)
        sum_x2y   = np.sum(x2y)
        sum_x1x2  = np.sum(x1x2)

        ### Means
        ybar = np.mean(y)
        x1bar = np.mean(x1)
        x2bar = np.mean(x2)

        ### Normalised
        n_x1_sq = sum_x1_sq - ( sum_x1**2 / n )
        n_x2_sq = sum_x2_sq - ( sum_x2**2 / n )
        n_x1y   = sum_x1y - ((sum_x1*sum_y)/n)
        n_x2y   = sum_x2y - ((sum_x2*sum_y)/n)
        n_x1x2   = sum_x1x2 - ((sum_x1*sum_x2)/n)

        b1 = ( (n_x2_sq*n_x1y) - (n_x1x2*n_x2y) )/( (n_x1_sq*n_x2_sq) - (n_x1x2**2) )
        b2 = ( (n_x1_sq*n_x2y) - (n_x1x2*n_x1y) )/( (n_x1_sq*n_x2_sq) - (n_x1x2**2) )
        b0 = ybar - (b1*x1bar) - (b2*x2bar)
        return b0, b1, b2

    def get_reg(self):
        x1 = np.linspace(0, max(self.x1), num=len(self.x1))
        x2 = np.linspace(0, max(self.x2), num=len(self.x2))
        b0, b1, b2 = self.get_coeffs()
        X1, X2 = np.meshgrid(x1, x2)
        Z = (b1*X1 + b2*X2 + b0)
        ax = plt.axes(projection='3d')
        ax.plot_surface(X1, X2, Z, color='r')
    
    def get_comp(self):
        ax = plt.axes(projection='3d')
        x1 = np.linspace(0, max(self.x1), num=len(self.x1))
        x2 = np.linspace(0, max(self.x2), num=len(self.x2))
        b0, b1, b2 = self.get_coeffs()
        X1, X2 = np.meshgrid(x1, x2)
        Z = (b1*X1 + b2*X2 + b0)
        ax.plot_surface(X1, X2, Z, color='r', alpha=0.8)
        plt.title("Scatter vs Regression")

        x1_arr = self.x1
        x2_arr = self.x2
        y  = self.y
        ax.scatter(x1_arr, x2_arr, y, color="b")

        
        
