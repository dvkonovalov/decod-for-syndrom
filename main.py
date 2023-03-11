import numpy

def create_generate_matrix(n, k):
    ed = numpy.eye(k, dtype=int)
    p = {}
    p[4] = numpy.array([[1,1], [0, 1]])
    p[15] = numpy.array([[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1],
                           [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1],
                          [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]])
    p[6] = numpy.array([[1, 0], [1, 2], [2, 1], [0, 2]])
    g = numpy.hstack((ed, p[n]))
    return g


def calc_h_matrix(g, f):
    (k, n) = g.shape
    p = g[0, k:n]
    for i in range(1, k):
        temp = g[i, k:n]
        p = numpy.vstack([p, temp])
    h = numpy.transpose(p)
    for i in range(n-k):
        for j in range(k):
            h[i, j] = (-1*h[i, j])%f
    h = numpy.vstack([h, numpy.eye(k, dtype=int)])
    return h


def bit_xor(string1, string2, f):
    answer = ''
    for i in range(len(string1)):
        answer += str((int(string1[i]) + int(string2[i]))%f)
    return answer

def innum(chislo, ss):
    n = ''
    k = ''
    while chislo > 0:
        n = n + str(chislo % ss)
        chislo = chislo // ss
    n = list(reversed(n))
    for j in range(len(n)):
        k += n[j]
    return k


def create_table_standard_location(matrix, f):
    (k, n) = matrix.shape
    mas = ['0'*n]
    lc = k+1
    for i in range(k):
        el = ''
        for j in range(n):
            el += str(matrix[i, j])
        mas.append(el)

    for i in range(f**k-k-1):
        pos = innum(lc, f)
        pos = '0'*(k-len(pos)) + pos
        summa = '0'*n
        for el in range(len(pos)):
            if pos[el]!='0':
                if pos[el]=='2':
                    temp = ''
                    for iii in mas[el+1]:
                        temp += str(int(iii)*2)
                    summa = bit_xor(summa, temp, f)
                else:
                    summa = bit_xor(summa, mas[el+1], f)
        lc += 1
        mas.append(summa)
    pos = '1' + '0'*(n-1)
    table = numpy.array(mas)
    for i in range(f**(n-k)-1):
        mas = [pos]
        for j in range(1,f**k):
            if (i==0):
                mas.append(bit_xor(pos, table[j], f))
            else:
                mas.append(bit_xor(pos, table[0, j], f))
        if (f==3 and pos.count('1')==1):
            pos = pos.replace('1', '2')
        else:
            pos = '0'+pos[:-1]
            pos = pos.replace('2', '1')
        table = numpy.vstack([table, numpy.array(mas)])
    return table


def decod_for_standard_location(table, element):
    (k, n) = table.shape
    for i in range(k):
        for j in range(n):
            if (table[i,j]==element):
                return table[i, 0]

fq = 3
a = create_generate_matrix(6, 4)
print(a)
b = create_table_standard_location(a, fq)
print(b, b.shape)


# for i in range(2**11):
#     for j in range(2**4):
#         if (b[j, i].count('1')==1):
#             print(j, i)
