import random
import time

import numpy


def random_vector(length, f):
    """
    Функция для создания рандомного вектора длины length
    :param length: длина вектора
    :param f: поле над которым строится вектор
    :return: массив в виде вектора
    """
    summa = 0
    mas = []
    for i in range(length):
        el = random.randint(0, f - 1)
        mas.append(el)
        if el>0:
            summa += 1

    while (summa<2 and length>2):
        summa = 0
        mas = []
        for i in range(length):
            el = random.randint(0, f - 1)
            mas.append(el)
            if el>0:
                summa += 1

    return mas

def create_generate_matrix(n, k, f):
    """
    Функция генерации порождающей канонической матрицы
    :param n: общее количество бит
    :param k:количество информационных бит
    :param f: поле над которым строится матрица
    :return:матрица-генератор
    """
    ed = numpy.eye(k, dtype=int)
    mas = []
    el = random_vector(n-k, f)
    p = numpy.array(el)
    mas.append(el)
    for i in range(k-1):
        el = random_vector(n-k, f)
        if not el in mas:
            p = numpy.vstack((p, numpy.array(el)))
            mas.append(el)
        else:
            while el in mas:
                el = random_vector(n-k, f)
            p = numpy.vstack((p, numpy.array(el)))
            mas.append(el)

    return numpy.hstack((ed, p))


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
    h = numpy.hstack([h, numpy.eye(n-k, dtype=int)])
    if transp==True:
        h = numpy.transpose(h)
    return h


def bit_xor(string1, string2, f):
    """
    Прибавление ошибки к правильному вектору
    :param string1: первое число в виде строки
    :param string2: второе число в виде строки
    :param f: поле над которым происходит сложение
    :return: строка с результатом сложения
    """
    answer = ''
    for i in range(len(string1)):
        answer += str((int(string1[i]) + int(string2[i]))%f)
    return answer

def bit_xor_revers(string1, string2, f):
    """
        Вычитание ошибки из вектора
        :param string1: первое число в виде строки
        :param string2: второе число в виде строки
        :param f: поле над которым происходит сложение
        :return: строка с результатом сложения
        """
    answer = ''
    for i in range(len(string1)):
        answer += str((int(string1[i]) - int(string2[i])) % f)
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
    if f==2:
        lc = 3
    else:
        lc = 2

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
        check = innum(lc, f)

        if check.count('0')+1==len(check) and check.count('2')<1:
            lc += 1
        mas.append(summa)

    # заполнили первую строку таблицы
    table = numpy.array(mas)
    (k, n) = matrix.shape
    if (n==8 and k==4):
        pos = []
        chislo = '1'+'0'*7
        for i in range(n):
            pos.append(chislo)
            pos.append(chislo.replace('1', '2'))
            chislo = '0' + chislo[:-1]
        chislo = '0' * 8
        ch = 0
        while chislo != '11000000':
            ch += 1
            chislo = innum(ch, 3)
            chislo = '0' * (8 - len(chislo)) + chislo
            if chislo.count('2') == 0 and chislo.count('1') == 2:
                pos.append(chislo)
                pos.append(chislo.replace('1', '2'))
        chislo = '0' * 8
        ch = 0
        while chislo!=('22' + '0'*(n-2)) and len(pos)<f**(n-k)-1:
            ch += 1
            chislo = innum(ch, 3)
            chislo = '0' * (8 - len(chislo)) + chislo
            if chislo.count('2') == 1 and chislo.count('1') == 1:
                pos.append(chislo)
                pos.append(chislo.replace('1', '2'))
    else:
        pos = []
        chislo = '1' + '0' * (n - 1)
        pos.append(chislo)
        for i in range(f**(n-k)-2):
            chislo = '0' + chislo[:-1]
            pos.append(chislo)
            if chislo.count('0')==n:
                chislo = '11' + '0'*(n-2)


    for i in range(f**(n-k)-1):
        mas = [pos[i]]
        for j in range(1,f**k):
            if (i==0):
                chislo = bit_xor(pos[i], table[j], f)
                mas.append(chislo)
            else:
                mas.append(bit_xor(pos[i], table[0, j], f))

        table = numpy.vstack([table, numpy.array(mas)])
    return table


def decod_for_standard_location(table, element):
    """
    Декодирование по стандартному расположению
    :param table: таблица стандартного расположения
    :param element: элемент (верный или который необходимо исправить)
    :return: исправленный элемент
    """
    stroka = ''
    for i in element:
        stroka += str(i)
    (k, n) = table.shape
    for i in range(k):
        for j in range(n):
            if (table[i,j]==stroka):
                return numpy.array(list(map(int, table[0, j])))

def decoding_by_syndrome(table, element, f, h):
    """
    Декодирование по синдрому
    :param table: таблица
    :param element: элемент который нужно исправить
    :param f: поле в котором мы работаем
    :param h: матрица H транспонированная
    :return: исправленный элемент
    """
    element1 = numpy.dot(element, h)
    rez_umn = ''
    for i in element1:
        rez_umn += str(i%f)
    rez_umn = '0'*(h.shape[1]-len(rez_umn)) + rez_umn

    element1 = ''
    for i in element:
        element1 += str(i)

    for i in range(table.shape[1]):
        if rez_umn==table[0, i]:

            element = bit_xor_revers(element1, table[1, i], f)
            element = numpy.array(list(map(int, element)))
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



def test_correct():
    """
    Тестирование построение генерирующих и проверочных матриц
    """

    """Начало тестов для кода (4, 2)"""
    print('Проверка матриц для (4, 2) кода')
    matrixg4 = create_generate_matrix(4, 2, 2)
    matrixh4 = calc_h_matrix(matrixg4, 2)
    correct = True
    for i in range(4):
        chislo = list(map(int, bin(i)[2:]))
        if len(chislo)<2:
            chislo = [0]*(2-len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrixg4)
        for j in range(4):
            chislo[j] = chislo[j]%2
        chislo = numpy.matmul(chislo, matrixh4)
        if (2*(chislo[1]%2) + chislo[0]%2 != 0):
            correct = False

    if (correct):
        print('Генерирующая и проверочная матрица для (4, 2) кода строятся верно!\n')
    else:
        print('Произошла ошибка! Матрицы построены неверно!\n')
    """Конец теста матриц (4, 2) кода"""


    """Начало тестов для кода (15, 11)"""
    print('Проверка матриц для (15, 11) кода')
    matrixg15 = create_generate_matrix(15, 11, 2)
    matrixh15 = calc_h_matrix(matrixg15, 2)
    correct = True
    for i in range(2**11):
        chislo = list(map(int, bin(i)[2:]))
        if len(chislo) < 11:
            chislo = [0] * (11 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrixg15)
        for j in range(15):
            chislo[j] = chislo[j] % 2
        chislo = numpy.matmul(chislo, matrixh15)
        summa = 0
        for j in range(4):
            summa = summa*2 + chislo[j] % 2
        if (summa != 0):
            correct = False

    if (correct):
        print('Генерирующая и проверочная матрица для (15, 11) кода строятся верно!\n')
    else:
        print('Произошла ошибка! Матрицы построены неверно!\n')
    """Конец теста матриц (15, 11) кода"""

    """Начало тестов для (6, 4, 3) кода"""
    print('Проверка матриц для (6, 4, 3) кода')
    matrixg6 = create_generate_matrix(6, 4, 3)
    matrixh6 = calc_h_matrix(matrixg6, 3)
    correct = True
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrixg6)
        for j in range(6):
            chislo[j] = chislo[j] % 3
        chislo = numpy.matmul(chislo, matrixh6)
        summa = 0
        for j in range(2):
            summa = summa * 3 + chislo[j] % 3
        if (summa != 0):
            correct = False

    if (correct):
        print('Генерирующая и проверочная матрица для (6, 4, 3) кода строятся верно!\n')
    else:
        print('Произошла ошибка! Матрицы построены неверно!')
    print()
    """Конец тестов для (6, 4, 3) кода"""

    """Начало тестов для (8, 4, 3) кода"""
    print('Проверка матриц для (8, 4, 3) кода')
    matrixg8 = create_generate_matrix(8, 4, 3)
    matrixh8 = calc_h_matrix(matrixg8, 3)
    correct = True
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrixg8)
        for j in range(8):
            chislo[j] = chislo[j] % 3
        chislo = numpy.matmul(chislo, matrixh8)
        summa = 0
        for j in range(4):
            summa = summa * 3 + chislo[j] % 3
        if (summa != 0):
            correct = False

    if (correct):
        print('Генерирующая и проверочная матрица для (8, 4, 3) кода строятся верно!\n')
    else:
        print('Произошла ошибка! Матрицы построены неверно!\n')
    """Конец тестов для (8, 4, 3) кода"""


def test_decod_standart_location():
    matrix4 = create_generate_matrix(4,2, 2)
    table4 = create_table_standard_location(matrix4, 2)
    matrix15 = create_generate_matrix(15,11, 2)
    table15 = create_table_standard_location(matrix15, 2)
    matrix6 = create_generate_matrix(6, 4, 3)
    table6 = create_table_standard_location(matrix6, 3)
    matrix8 = create_generate_matrix(8, 4, 3)
    table8 = create_table_standard_location(matrix8, 3)


    """Старт тестов"""
    # (4,2) код
    t0 = time.time()
    correct = True
    summa = 0
    for i in range(2**2):
        chislo = list(map(int, bin(i)[2:]))
        if len(chislo) < 2:
            chislo = [0] * (2 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix4)
        for i in range(len(chislo)):
            chislo[i] = (chislo[i])%2
        pr = numpy.copy(chislo)
        pos = random.randint(0, 3)
        chislo[pos] = (chislo[pos]+1)%2

        chislo = decod_for_standard_location(table4, chislo)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(f'Программа успешно исправляет все однократные ошибки в (4, 2) коде. Количество тестов - {2**2}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (4, 2) коде c 1 ошибкой. Количество тестов - {2**2}, из них неправильно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')

    # (15, 11) код
    correct = True
    summa = 0
    t0 = time.time()
    for i in range(2**11):
        chislo = list(map(int, bin(i)[2:]))
        if len(chislo) < 11:
            chislo = [0] * (11 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix15)
        for j in range(15):
            chislo[j] = (chislo[j])%2
        pr = numpy.copy(chislo)
        pos = random.randint(0, 14)
        chislo[pos] = (chislo[pos]+1) % 2

        chislo = decod_for_standard_location(table15, chislo)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(f'Программа успешно исправляет все однократные ошибки в (15, 11) коде. Количество тестов - {2**11}. Время работы программы - ', round(t1-t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (15, 11) коде c 1 ошибкой. Количество ошибок {summa}')


    # (6, 4, 3) код
    correct = True
    t0 = time.time()
    summa = 0
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix6)
        for j in range(6):
            chislo[j] = (chislo[j]) % 3
        pr = numpy.copy(chislo)
        pos = random.randint(0, 5)
        chislo[pos] = (chislo[pos] + 1) % 3

        chislo = decod_for_standard_location(table6, chislo)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(f'Программа успешно исправляет все однократные ошибки в (6, 4) коде. Количество тестов - {3**4}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (6, 4) коде c 1 ошибкой. Количество тестов - {3**4}, из них неверно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')


    # (8, 4, 3) код
    correct = True
    summa = 0
    t0 = time.time()
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix8)
        for j in range(8):
            chislo[j] = (chislo[j]) % 3
        pr = numpy.copy(chislo)
        pos = random.randint(0, 7)
        chislo[pos] = (chislo[pos] + 1) % 3

        chislo = decod_for_standard_location(table8, chislo)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(f'Программа успешно исправляет все однократные ошибки в (8, 4) коде. Количество тестов - {3**4} Время работы программы - ',
              round(t1 - t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (8, 4) коде c 1 ошибкой. Количество тестов - {3**4}, из них неверно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')



def test_decod_syndrom():
    """
    Тестирование декодирования по синдрому
    """
    matrix4 = create_generate_matrix(4, 2, 2)
    table_standart_location = create_table_standard_location(matrix4, 2)
    matrix_h_4 = calc_h_matrix(matrix4, 2)
    table4 = get_table_syndrom(table_standart_location, matrix_h_4)

    matrix15 = create_generate_matrix(15, 11, 2)
    table_standart_location = create_table_standard_location(matrix15, 2)
    matrix_h_15 = calc_h_matrix(matrix15, 2)
    table15 = get_table_syndrom(table_standart_location, matrix_h_15)

    matrix6 = create_generate_matrix(6, 4, 3)
    table_standart_location = create_table_standard_location(matrix6, 3)
    matrix_h_6 = calc_h_matrix(matrix6, 3)
    table6 = get_table_syndrom(table_standart_location, matrix_h_6)

    matrix8 = create_generate_matrix(8, 4, 3)
    table_standart_location = create_table_standard_location(matrix8, 3)
    matrix_h_8 = calc_h_matrix(matrix8, 3)
    table8 = get_table_syndrom(table_standart_location, matrix_h_8)

    """Старт тестов"""
    # (4,2) код
    t0 = time.time()
    correct = True
    summa = 0
    for i in range(2 ** 2):
        chislo = list(map(int, bin(i)[2:]))
        if len(chislo) < 2:
            chislo = [0] * (2 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix4)
        for i in range(len(chislo)):
            chislo[i] = (chislo[i]) % 2

        pr = numpy.copy(chislo)
        pos = random.randint(0, 3)
        chislo[pos] = (chislo[pos] + 1) % 2

        chislo = decoding_by_syndrome(table4, chislo, 2, matrix_h_4)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(
            f'Программа успешно исправляет все однократные ошибки в (4, 2) коде. Количество тестов - {2 ** 2}. Время работы программы - ',
            round(t1 - t0, 3), ' секунд')
    else:
        print(
            f'Произошла ошибка декодирования в (4, 2) коде c 1 ошибкой. Количество тестов - {2 ** 2}, из них неправильно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')

    # (15, 11) код
    correct = True
    t0 = time.time()
    for i in range(2 ** 11):

        chislo = list(map(int, bin(i)[2:]))
        if len(chislo) < 11:
            chislo = [0] * (11 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix15)
        for j in range(15):
            chislo[j] = (chislo[j]) % 2

        pr = numpy.copy(chislo)
        pos = random.randint(0, 14)
        chislo[pos] = (chislo[pos] + 1) % 2

        chislo = decoding_by_syndrome(table15, chislo, 2, matrix_h_15)
        if not numpy.array_equal(chislo, pr):
            correct = False
    t1 = time.time()
    if correct:
        print(
            f'Программа успешно исправляет все однократные ошибки в (15, 11) коде. Количество тестов - {2 ** 11}. Время работы программы - ',
            round(t1 - t0, 3), ' секунд')
    else:
        print('Произошла ошибка декодирования в (15, 11) коде c 1 ошибкой')

    # (6, 4, 3) код
    correct = True
    t0 = time.time()
    summa = 0
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix6)
        for j in range(6):
            chislo[j] = (chislo[j]) % 3

        pr = numpy.copy(chislo)
        pos = random.randint(0, 5)
        chislo[pos] = (chislo[pos] + 1) % 3

        chislo = decoding_by_syndrome(table6, chislo, 3, matrix_h_6)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(
            f'Программа успешно исправляет все однократные ошибки в (6, 4) коде. Количество тестов - {3 ** 4}. Время работы программы - ',
            round(t1 - t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (6, 4) коде c 1 ошибкой. Количество тестов - {3 ** 4}, из них неверно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')

    # (8, 4, 3) код
    correct = True
    summa = 0
    t0 = time.time()
    for i in range(3 ** 4):
        chislo = list(map(int, innum(i, 3)))
        if len(chislo) < 4:
            chislo = [0] * (4 - len(chislo)) + chislo
        chislo = numpy.matmul(numpy.array(chislo), matrix8)
        for j in range(8):
            chislo[j] = (chislo[j]) % 3

        pr = numpy.copy(chislo)
        pos = random.randint(0, 7)
        chislo[pos] = (chislo[pos] + 1) % 3

        chislo = decoding_by_syndrome(table8, chislo, 3, matrix_h_8)
        if not numpy.array_equal(chislo, pr):
            correct = False
            summa += 1
    t1 = time.time()
    if correct:
        print(
            f'Программа успешно исправляет все однократные ошибки в (8, 4) коде. Количество тестов - {3 ** 4} Время работы программы - ',
            round(t1 - t0, 3), ' секунд')
    else:
        print(f'Произошла ошибка декодирования в (8, 4) коде c 1 ошибкой. Количество тестов - {3 ** 4}, из них неверно - {summa}. Время работы программы - ',
              round(t1 - t0, 3), ' секунд')




test_correct()
print('-----------------Декодирование по стандартному расположению-----------------')
test_decod_standart_location()
print('-----------------Декодирование по синдрому-----------------')
test_decod_syndrom()
