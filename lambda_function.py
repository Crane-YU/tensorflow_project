def g(n):
    return lambda x: x / n


k = g(1)  # n=1
print(k(2))  # x=2

m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
n = [[1, 1, 1], [2, 2, 3], [3, 3, 3]]
y = [x==y for x, y in zip(m, n)]
print(y)