import io, json, random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

histogram1Star = []
histogram2Star = []
histogram3Star = []
histogram4Star = []
histogram5Star = []

nSamples = 100

with open("data/output/histogram_1star.json") as oneStarFile:
    data = json.load(oneStarFile)
    for i, item in enumerate(data):
        if i > nSamples:
            break
        histogram1Star.append(item["histogram"])

with open("data/output/histogram_2star.json") as twoStarFile:
    data = json.load(twoStarFile)
    for i, item in enumerate(data):
        if i > nSamples:
            break
        histogram2Star.append(item["histogram"])

with open("data/output/histogram_3star.json") as threeStarFile:
    data = json.load(threeStarFile)
    for i, item in enumerate(data):
        if i > nSamples:
            break
        histogram3Star.append(item["histogram"])

with open("data/output/histogram_4star.json") as fourStarFile:
    data = json.load(fourStarFile)
    for i, item in enumerate(data):
        if i > nSamples:
            break
        histogram4Star.append(item["histogram"])

with open("data/output/histogram_5star.json") as fiveStarFile:
    data = json.load(fiveStarFile)
    for i, item in enumerate(data):
        if i > nSamples:
            break
        histogram5Star.append(item["histogram"])

matrix1Star = pairwise_distances(histogram1Star, Y=None, metric='cosine', n_jobs=1)
matrix2Star = pairwise_distances(histogram2Star, Y=None, metric='cosine', n_jobs=1)
matrix3Star = pairwise_distances(histogram3Star, Y=None, metric='cosine', n_jobs=1)
matrix4Star = pairwise_distances(histogram4Star, Y=None, metric='cosine', n_jobs=1)
matrix5Star = pairwise_distances(histogram5Star, Y=None, metric='cosine', n_jobs=1)

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

plt.plot(x_array1Star, y_array1Star, 'ro', markersize=2.0)
plt.plot(x_array2Star, y_array2Star, 'ro', markersize=2.0)
plt.plot(x_array3Star, y_array3Star, 'ro', markersize=2.0)
plt.plot(x_array4Star, y_array4Star, 'ro', markersize=2.0)
plt.plot(x_array5Star, y_array5Star, 'ro', markersize=2.0)
plt.axis([0, 6, -1, 2])


plt.axvline(x=1, ymin=0, ymax = 1, linewidth=1, color='r')
plt.axvline(x=2, ymin=0, ymax = 1, linewidth=1, color='r')
plt.axvline(x=3, ymin=0, ymax = 1, linewidth=1, color='r')
plt.axvline(x=4, ymin=0, ymax = 1, linewidth=1, color='r')
plt.axvline(x=5, ymin=0, ymax = 1, linewidth=1, color='r')

plt.show()


