import numpy as np
import matplotlib.pyplot as plt

class LinReg:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def __coeffs(self):
        x = self.x
        y = self.y

        x2 = x**2
        xy = x*y
        n = len(x)

        a = (sum(y)*sum(x2) - sum(x)*sum(xy))/(n*sum(x2) - sum(x)**2)
        b = (n*sum(xy) - sum(x)*sum(y))/(n*sum(x2) - sum(x)**2)

        return a, b

    def predict(self, y):
        a, b = self.__coeffs()
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
        a, b = self.__coeffs()
        x = np.linspace(0, max(self.x))
        y = a + b*x
        plt.plot(x, y, 'r-', label="Line of Best Fit")
        plt.title("Linear Regression")
        plt.xlim(0, (max(x)+1))
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.legend()
        plt.show()

    def get_comp(self):
        a, b = self.__coeffs()
        x = np.linspace(0, max(self.x))
        y = a + b*x
        plt.plot(x, y, 'r-', label="Line of Best Fit")
        plt.scatter(self.x, self.y, color="orange", alpha=0.7)
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
        fig = plt.figure(figsize=(7,6))
        fig.suptitle("Scatterplots", fontsize=18)

        ax = fig.add_subplot(2,2,1, xlabel="X1", ylabel="X2", title="X1 and X2")
        ax.scatter(x1,x2)
        

        ax = fig.add_subplot(2,2,2, xlabel="X1", ylabel="Y", title="X1 and Y")
        ax.scatter(x1,y)

        ax = fig.add_subplot(2,2,3, xlabel="X2", ylabel="Y", title="X2 and Y")
        ax.scatter(x2,y)

        ax = fig.add_subplot(2,2,4, projection="3d", xlabel="X1", ylabel="X2", zlabel="Y", title="X1, X2 and Y")
        ax.scatter(x1,x2,y)
        plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
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
        ax = plt.axes(projection='3d', xlabel="X1", ylabel="X2", zlabel="Y")
        plt.title("Regression Plane 3D")
        ax.plot_surface(X1, X2, Z, color='r')
    
    def get_comp(self):
        ax = plt.axes(projection='3d', xlabel="X1", ylabel="X2", zlabel="Y")
        x1 = np.linspace(0, max(self.x1), num=len(self.x1))
        x2 = np.linspace(0, max(self.x2), num=len(self.x2))
        b0, b1, b2 = self.get_coeffs()
        X1, X2 = np.meshgrid(x1, x2)
        Z = (b1*X1 + b2*X2 + b0)
        ax.plot_surface(X1, X2, Z, color='r', alpha=0.8)
        plt.title("Scatter vs Regression", fontsize=18)

        x1_arr = self.x1
        x2_arr = self.x2
        y  = self.y
        ax.scatter(x1_arr, x2_arr, y, color="b")

    def __corcof(self, x, y):

        n = len(x)
        xy = x*y
        x2 = x**2
        y2 = y**2

        r = (n*sum(xy) - sum(x)*sum(y)) / \
            np.sqrt((n*sum(x2) - sum(x)**2)*(n*sum(y2) - sum(y)**2))

        return r

    def get_corcof(self):
        """
        Gets the R^2 value
        """
        x1 = self.x1
        x2 = self.x2
        y  = self.y

        r_x1y  = self.__corcof(x1, y)
        r_x2y  = self.__corcof(x2, y)
        r_x1x2 = self.__corcof(x1,x2)

        r_x1y_sq  = r_x1y**2
        r_x2y_sq  = r_x2y**2
        r_x1x2_sq = r_x1x2**2

        R = np.sqrt( ( r_x1y_sq + r_x2y_sq - (2*r_x1y*r_x2y*r_x1x2) )  / ( 1 - r_x1x2_sq ) )

        return R      


def Lagrange(x: list, y: list, x0: float):
    """
    Plots the Lagrange polynomial and returns the value at x0

    @params:
        x: list
        y: list
        x0: float
    """  
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    def __calc(xp):
        yp = 0
        for i in range(n):
            p = 1
            for j in range(n):
                if i != j:
                    p = p*(xp - x[j])/(x[i] - x[j])
            yp = yp + p * y[i]
        return yp

    xp = np.linspace(0, max(x)+5)
    y_plot = __calc(xp)
    y0 = __calc(x0)
    plt.plot(xp, y_plot, label="Interpolated Line")
    plt.plot(x, y, 'ro', label="Data Points")
    plt.plot(x0, y0, 'go', label="Interpolated Point")
    plt.title("Lagrange Interpolation", fontsize=18)
    plt.legend()
    plt.xlabel("X Values")
    plt.ylabel("Y Values")

    for i_x, i_y in zip(x, y):
        plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))

    plt.text(x0, y0, f"({x0}, {y0:.2f})", fontweight='bold')
    print("At x0:", y0)