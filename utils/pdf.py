import matplotlib.pyplot as plt
import numpy as np

class utilsClass:

    def pdfFunction():
        sample = np.random.normal(size=1000)

        plt.hist(sample, bins=20)
        plt.show()

        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        print(sample_mean)
        print(sample_std)

    def generateGaussians():
        
        x = np.linspace(-3,3,120)
        for mu, sigma in [(-1,1),(0,2),(2,3)]:
            plt.plot(x, np.exp(-np.power(x-mu, 2.) / (2 * np.power(sigma, 2))))

        plt.show()


if __name__ == "__main__":

    utilsClass.generateGaussians()
