import numpy

def create_generate_matrix(n, k):
    """
    Функция генерации порождающей канонической матрицы
    :param n: общее количество бит
    :param k:количество информационных бит
    :return:матрица-генератор
    """
    ed = numpy.eye(k, dtype=int)
    p = {}
    p[4] = numpy.array([[0,1], [1, 1]])
    p[15] = numpy.array([[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1],
                           [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 1],
                          [0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0]])
    p[6] = numpy.array([[1, 0], [1, 2], [2, 1], [0, 2]])
    g = numpy.hstack((ed, p[n]))
    return g


def calc_h_matrix(g, f, transp = True):
    """
    Функция для получения проверочной матрицы
    :param g:матрица-генератор
    :param f:поле над которым заданы матрицы
    :param transp: True - транспонированная, False - не транспонированная
    :return:
    """
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
    if transp==False:
        h = numpy.transpose(h)
    return h


def bit_xor(string1, string2, f):
    """
    Побитовое сложение
    :param string1: первое число в виде строки
    :param string2: второе число в виде строки
    :param f: поле над которым происходит сложение
    :return: строка с результатом сложения
    """
    answer = ''
    for i in range(len(string1)):
        answer += str((int(string1[i]) + int(string2[i]))%f)
    return answer

def innum(chislo, ss):
    """
    Перевод из десятичной СС в ss-систему счисления
    :param chislo: число в 10 СС
    :param ss: необходимая СС
    :return: число в ss-ной СС
    """
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
    """
    Создание таблицы стандартного расположения
    :param matrix: матрица-генератор
    :param f: поле используемое в матрице
    :return: таблица стандартного расположения
    """
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
    """
    Декодирование по стандартному расположению
    :param table: таблица стандартного расположения
    :param element: элемент (верный или который необходимо исправить)
    :return: исправленный элемент
    """
    (k, n) = table.shape
    for i in range(k):
        for j in range(n):
            if (table[i,j]==element):
                return table[0, j]

def decoding_by_syndrome(table, element, f, h):
    """
    Декодирование по синдрому
    :param table: таблица
    :param element: элемент который нужно исправить
    :param f: поле в котором мы работаем
    :param h: матрица H
    :return: исправленный элемент
    """
    element = [int(i) for i in element]
    element1 = numpy.dot(h, element)
    element = ''
    for i in element1:
        element += str(i%f)
    for i in range(table.shape[1]):
        if element==table[0, i]:
            element = bit_xor(element, table[1, i], f)
            break
    return element

def get_table_syndrom(table_standard_location, matrix_ht):
    """
    Создание таблицы для декодирования по синдрому
    :param table_standard_location: таблица стандартного расположения
    :param matrix_ht: транспонированная матрица H
    :return: таблица синдромов и ошибок
    """
    leader = []
    syndrome = []
    for i in table_standard_location:
        leader.append(i[0])
        element = [int(i) for i in i[0]]
        element = numpy.array(element)
        s = numpy.dot(element, matrix_ht)
        ns = ''
        for j in s:
            ns += str(j)
        syndrome.append(ns)
    return numpy.array([syndrome,leader])




fq = 2
a = create_generate_matrix(4, 2)
h = calc_h_matrix(a, fq)
hn = calc_h_matrix(a, fq, False)
b = create_table_standard_location(a, fq)
d = get_table_syndrom(b, h)


