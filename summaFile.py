
def sumFile():
    for numb in range(1,40):
        path1 = "Признаки\\1\\Депрессия.txt"
        path2 = "Признаки\\2\\Депрессия.txt"
        path3 = "Признаки\\summa1and2.txt"

        f1 = open(path1, 'r')
        f2 = open(path2, 'r')
        f3 = open(path3, 'w')

        k = 0
        for i in range(121):
            k = k + 1
            s1 = f1.readline()
            s1 = s1[0:-10]

            s2 = f2.readline()

            s3 = str(s1) + ' ' + str(s2)
            f3.write(s3)

        f1.close()
        f2.close()
        f3.close()
sumFile()