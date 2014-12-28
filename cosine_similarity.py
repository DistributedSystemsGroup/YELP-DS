import io, json, random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib.backends.backend_pdf import PdfPages
    
def drawCOSFigure(features, output):    
    
    histogram1Star = []
    histogram2Star = []
    histogram3Star = []
    histogram4Star = []
    histogram5Star = []
    
    nSamples = 100
    metricType = 'cosine'
    
    with open("data/output/cosine_similarity/histogram_1star.json") as oneStarFile:
        data = json.load(oneStarFile)
        for i, item in enumerate(data):
            if i > nSamples:
                break
            histogram1Star.append(list(item["histogram"][i] for i in features))
    
    with open("data/output/cosine_similarity/histogram_2star.json") as twoStarFile:
        data = json.load(twoStarFile)
        for i, item in enumerate(data):
            if i > nSamples:
                break
            histogram2Star.append(list(item["histogram"][i] for i in features))
    
    with open("data/output/cosine_similarity/histogram_3star.json") as threeStarFile:
        data = json.load(threeStarFile)
        for i, item in enumerate(data):
            if i > nSamples:
                break
            histogram3Star.append(list(item["histogram"][i] for i in features))
    
    with open("data/output/cosine_similarity/histogram_4star.json") as fourStarFile:
        data = json.load(fourStarFile)
        for i, item in enumerate(data):
            if i > nSamples:
                break
            histogram4Star.append(list(item["histogram"][i] for i in features))
    
    with open("data/output/cosine_similarity/histogram_5star.json") as fiveStarFile:
        data = json.load(fiveStarFile)
        for i, item in enumerate(data):
            if i > nSamples:
                break
            histogram5Star.append(list(item["histogram"][i] for i in features))
    
    matrix1Star = pairwise_distances(histogram1Star, Y=None, metric=metricType, n_jobs=1)
    matrix2Star = pairwise_distances(histogram2Star, Y=None, metric=metricType, n_jobs=1)
    matrix3Star = pairwise_distances(histogram3Star, Y=None, metric=metricType, n_jobs=1)
    matrix4Star = pairwise_distances(histogram4Star, Y=None, metric=metricType, n_jobs=1)
    matrix5Star = pairwise_distances(histogram5Star, Y=None, metric=metricType, n_jobs=1)
    
    length = nSamples
    x_array1Star = []
    y_array1Star = []
    
    x_array2Star = []
    y_array2Star = []
    
    x_array3Star = []
    y_array3Star = []
    
    x_array4Star = []
    y_array4Star = []
    
    x_array5Star = []
    y_array5Star = []
    left = -0.49
    right = 0.49
    for i in xrange(0, length):
        for j in xrange(i, length):
            x = 0.5 + random.uniform(left, right)
            y = 1- matrix1Star[i][j]
            x_array1Star.append(x)
            y_array1Star.append(y)
    
    for i in xrange(0, length):
        for j in xrange(i, length):
            x = 1.5 + random.uniform(left, right)
            y = 1- matrix2Star[i][j]
            x_array2Star.append(x)
            y_array2Star.append(y)
    
    for i in xrange(0, length):
        for j in xrange(i, length):
            x = 2.5 + random.uniform(left, right)
            y = 1- matrix3Star[i][j]
            x_array3Star.append(x)
            y_array3Star.append(y)
    
    for i in xrange(0, length):
        for j in xrange(i, length):
            x = 3.5 + random.uniform(left, right)
            y = 1- matrix4Star[i][j]
            x_array4Star.append(x)
            y_array4Star.append(y)
    
    for i in xrange(0, length):
        for j in xrange(i, length):
            x = 4.5 + random.uniform(left, right)
            y = 1- matrix5Star[i][j]
            x_array5Star.append(x)
            y_array5Star.append(y)
    pp = PdfPages(output)
    plt.figure()
    plt.plot(x_array1Star, y_array1Star, 'ro', markersize=2.0)
    plt.plot(x_array2Star, y_array2Star, 'ro', markersize=2.0)
    plt.plot(x_array3Star, y_array3Star, 'ro', markersize=2.0)
    plt.plot(x_array4Star, y_array4Star, 'ro', markersize=2.0)
    plt.plot(x_array5Star, y_array5Star, 'ro', markersize=2.0)
    plt.axis([0, 5, 0, 1])
    
    
    plt.axvline(x=1, ymin=0, ymax = 1, linewidth=1, color='r')
    plt.axvline(x=2, ymin=0, ymax = 1, linewidth=1, color='r')
    plt.axvline(x=3, ymin=0, ymax = 1, linewidth=1, color='r')
    plt.axvline(x=4, ymin=0, ymax = 1, linewidth=1, color='r')
    plt.axvline(x=5, ymin=0, ymax = 1, linewidth=1, color='r')
    
    #plt.show()
    plt.savefig(pp, format='pdf')
    pp.close()

def main():
    features = []
    
    features1 = [0, 1, 2, 3, 4, 5, 8]
    features2 = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
    features3 = [0, 1, 2, 3, 4, 5, 8, 14, 15, 16, 17, 18, 19]
    features4 = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features5 = [6, 7, 8]
    features6 = [6, 7, 8, 9, 10, 11]
    features7 = [6, 7, 8, 14, 15, 16, 17, 18, 19]
    features8 = [6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features9 = [12, 13, 8]
    features10 = [12, 13, 8, 9, 10, 11]
    features11 = [12, 13, 8, 14, 15, 16, 17, 18, 19]
    features12 = [12, 13, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
    features13 = [20, 21, 22, 23, 24]
    features14 = [20, 21, 22, 23, 24, 9, 10, 11]
    features15 = [20, 21, 22, 23, 24, 14, 15, 16, 17, 18, 19]
    
    features.append(features1)
    features.append(features2)
    features.append(features3)
    features.append(features4)
    features.append(features5)
    features.append(features6)
    features.append(features7)
    features.append(features8)
    features.append(features9)
    features.append(features10)
    features.append(features11)
    features.append(features12)
    features.append(features13)
    features.append(features14)
    features.append(features15)

    print(features)

    for i in xrange(1, 16):
        print(i)
        output = 'report/refs/COS/multipage' + str(i) + '.pdf'
        drawCOSFigure(features[i-1], output)
    
if __name__ == "__main__":
    main()
    
    
