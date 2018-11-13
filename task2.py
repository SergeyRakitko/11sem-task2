import math
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import symbols, Matrix, solve

# создаем подкаталог results для сохранения фото результатов
current_directory = os.path.dirname(__file__)
folder_name = "results"
current_directory += "/" + folder_name
if not os.path.exists(current_directory):
    os.makedirs(current_directory)

# значения параметров модели
k1Value = 0.12
k1rValueList = [0.001, 0.005, 0.01, 0.015, 0.02]
k2Value = 2.5
k3Value = 0.0032
k3rValueList = [0.0005, 0.001, 0.002, 0.003, 0.004]

# параметры для отрисовки
fig, ax = plt.subplots()

# описываем уравнения системы
x, y = symbols("x y")
k1, k1r, k2, k3, k3r = symbols("k1 k1r k2 k3 k3r")
z = 1 - x - 2 * y
f1 = k1 * z - k1r * x - k2 * z ** 2 * x
f2 = k3 * z ** 2 - k3r * y

# матрица Якоби
A = Matrix([f1, f2])
varVector = Matrix([x, y])
jacA = A.jacobian(varVector)

# опеределитель и след матрицы Якоби
determinant = jacA.det()
trace = jacA.trace()

# решаем систему
staticSolution = solve([f1, f2], x, k2)

####################################################################################################################
# однопараметрический анализ по параметру k1r
print("Однопараметрический анализ по параметру k1r начат ...")

# выбираем решение, подставляем известные параметры
solNumber = 0  # 0, 1, второе решение не лежит в ОДЗ
sol = [staticSolution[solNumber][0].subs({k1: k1Value, k3: k3Value, k3r: k3rValueList[1]}),
       staticSolution[solNumber][1].subs({k1: k1Value, k3: k3Value, k3r: k3rValueList[1]})]

meshSize = 100  # плотность сетки по переменной 'y'
yGrid = np.linspace(0, 0.5, meshSize, dtype=float)

# для всех значений параметра k1r
for j in range(len(k1rValueList)):
    # списки координат стационарных решений
    yCoord = []
    k2Coord = []
    xCoord = []
    # списки координат седло-узловых биффуркаций
    horBif1List = []
    verBif1List = []
    # списки координат биффуркаций Андронова-Хопфа
    horBif2List = []
    verBif2List = []
    for i in range(meshSize):
        # рассматриваемые точки
        yNew = yGrid[i]
        k2New = sol[1].subs({y: yNew, k1r: k1rValueList[j]})
        xNew = sol[0].subs({y: yNew, k1r: k1rValueList[j]})

        # проверяем ОДЗ переменных
        if not (math.isnan(xNew) or math.isnan(yNew) or math.isnan(k2New)):
            if (xNew >= 0) and (xNew <= 1) and (xNew + 2 * yNew >= 0) and (xNew + 2 * yNew <= 1):
                # добавляем в списки координат
                yCoord.append(yNew)
                xCoord.append(xNew)
                k2Coord.append(k2New)

    # исследуем на биффуркации
    detPrev = determinant.subs({y: yCoord[0], k1r: k1rValueList[j], x: xCoord[0], k2: k2Coord[0], k1: k1Value,
                                k3: k3Value, k3r: k3rValueList[1]})
    detPrev = detPrev.evalf(7)
    tracePrev = trace.subs({y: yCoord[0], k1r: k1rValueList[j], x: xCoord[0], k2: k2Coord[0], k1: k1Value,
                            k3: k3Value, k3r: k3rValueList[1]})
    tracePrev = tracePrev.evalf(7)
    for k in range(len(xCoord) - 1):
        detNext = determinant.subs({y: yCoord[k + 1], k1r: k1rValueList[j], x: xCoord[k + 1],
                                    k2: k2Coord[k + 1], k1: k1Value, k3: k3Value, k3r: k3rValueList[1]})
        detNext = detNext.evalf(7)
        # седло узловая бифуркация
        if detPrev * detNext <= 0:
            horBif1List.append(k2Coord[k + 1])
            verBif1List.append(xCoord[k + 1])
            horBif1List.append(k2Coord[k + 1])
            verBif1List.append(yCoord[k + 1])
        detPrev = detNext

        traceNext = trace.subs({y: yCoord[k + 1], k1r: k1rValueList[j], x: xCoord[k + 1],
                                k2: k2Coord[k + 1], k1: k1Value, k3: k3Value, k3r: k3rValueList[1]})
        traceNext = traceNext.evalf(7)
        # бифуркация Андронова-Хопфа
        if tracePrev * traceNext <= 0:
            horBif2List.append(k2Coord[k + 1])
            verBif2List.append(xCoord[k + 1])
            horBif2List.append(k2Coord[k + 1])
            verBif2List.append(yCoord[k + 1])
        tracePrev = traceNext

    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='Седло-узловые')
    ax.plot(horBif2List, verBif2List, 'ro', label='Андронова-Хопфа', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k1r=' + str(k1rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')

    # сохраняем
    fig.savefig(folder_name + "/k1r_" + str(j + 1) + ".png")

print("Однопараметрический анализ по параметру k1r окончен")

##########################################################
# однопараметрический анализ по параметру k3r
print("Однопараметрический анализ по параметру k3r начат ...")

# выбираем решение, подставляем известные параметры
solNumber = 0  # 0, 1, второе решение не лежит в ОДЗ
sol = [staticSolution[solNumber][0].subs({k1: k1Value, k3: k3Value, k1r: k1rValueList[2]}),
       staticSolution[solNumber][1].subs({k1: k1Value, k3: k3Value, k1r: k1rValueList[2]})]

meshSize = 100  # плотность сетки по переменной 'y'
yGrid = np.linspace(0, 0.5, meshSize, dtype=float)

# для всех значений параметра k3r
for j in range(len(k3rValueList)):
    # списки координат стационарных решений
    yCoord = []
    k2Coord = []
    xCoord = []
    # списки координат седло-узловых биффуркаций
    horBif1List = []
    verBif1List = []
    # списки координат биффуркаций Андронова-Хопфа
    horBif2List = []
    verBif2List = []

    for i in range(meshSize):
        # рассматриваемые точки
        yNew = yGrid[i]
        k2New = sol[1].subs({y: yNew, k3r: k3rValueList[j]})
        xNew = sol[0].subs({y: yNew, k3r: k3rValueList[j]})

        # проверяем ОДЗ переменных
        if not (math.isnan(xNew) or math.isnan(yNew) or math.isnan(k2New)):
            if (xNew >= 0) and (xNew <= 1) and (xNew + 2 * yNew >= 0) and (xNew + 2 * yNew <= 1):
                # добавляем в списки координат
                yCoord.append(yNew)
                xCoord.append(xNew)
                k2Coord.append(k2New)

    # исследуем на биффуркации
    detPrev = determinant.subs({y: yCoord[0], k1r: k1rValueList[2], x: xCoord[0], k2: k2Coord[0], k1: k1Value,
                                k3: k3Value, k3r: k3rValueList[j]})
    detPrev = detPrev.evalf(7)
    tracePrev = trace.subs({y: yCoord[0], k1r: k1rValueList[2], x: xCoord[0], k2: k2Coord[0], k1: k1Value,
                            k3: k3Value, k3r: k3rValueList[j]})
    tracePrev = tracePrev.evalf(7)
    for k in range(len(xCoord) - 1):
        detNext = determinant.subs({y: yCoord[k + 1], k1r: k1rValueList[2], x: xCoord[k + 1],
                                    k2: k2Coord[k + 1], k1: k1Value, k3: k3Value, k3r: k3rValueList[j]})
        detNext = detNext.evalf(7)
        # седло узловая бифуркация
        if detPrev * detNext <= 0:
            horBif1List.append(k2Coord[k + 1])
            verBif1List.append(xCoord[k + 1])
            horBif1List.append(k2Coord[k + 1])
            verBif1List.append(yCoord[k + 1])
        detPrev = detNext

        traceNext = trace.subs({y: yCoord[k + 1], k1r: k1rValueList[2], x: xCoord[k + 1],
                                k2: k2Coord[k + 1], k1: k1Value, k3: k3Value, k3r: k3rValueList[j]})
        traceNext = traceNext.evalf(7)
        # бифуркация Андронова-Хопфа
        if tracePrev * traceNext <= 0:
            horBif2List.append(k2Coord[k + 1])
            verBif2List.append(xCoord[k + 1])
            horBif2List.append(k2Coord[k + 1])
            verBif2List.append(yCoord[k + 1])
        tracePrev = traceNext

    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='Седло-узловые')
    ax.plot(horBif2List, verBif2List, 'ro', label='Андронова-Хопфа', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k3r=' + str(k3rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')

    # сохраняем
    fig.savefig(folder_name + "/k3r_" + str(j+1) + ".png")

print("Однопараметрический анализ по параметру k3r окончен")


####################################################################################################################
# двухпараметрический анализ по параметрам k1, k2
print("Двухпараметрический анализ по параметрам k1, k2 начат ...")

# ищем линии кратности
# подставляем известные параметры
f1Count1 = f1.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
f2Count1 = f2.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
determinantCount1 = determinant.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
# решаем расширенную систему
complexSolution1 = solve([f1Count1, f2Count1, determinantCount1], y, k1, k2)
sol1 = complexSolution1[0]

# ищем линии нейтральности
# подставляем известные параметры
f1Count2 = f1.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
f2Count2 = f2.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
traceCount2 = trace.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
# решаем расширенную систему
complexSolution2 = solve([f1Count2, f2Count2, traceCount2], y, k1, k2)
sol2 = complexSolution2[0]

# сетка по переменной 'x'
xGrid = np.linspace(0, 0.9773, 100, dtype=float)
k1CoordK = []
k2CoordK = []
k1CoordN = []
k2CoordN = []

for i in range(len(xGrid)):
    # линии кратности
    # вычисляем значение k1
    k1New = sol1[1].subs(x, xGrid[i])
    k1New = k1New.evalf(7)
    # вычисляем значение k2
    k2New = sol1[2].subs(x, xGrid[i])
    k2New = k2New.evalf(7)
    # добавляем координаты в пределах [0, 10] x [0, 10]
    if (k1New >= 0) and (k1New <= 10) and (k2New >= 0) and (k2New <= 10):
        k1CoordK.append(k1New)
        k2CoordK.append(k2New)
    # линии нейтральности

    # вычисляем значение k1
    k1New = sol2[1].subs(x, xGrid[i])
    k1New = k1New.evalf(7)
    # вычисляем значение k2
    k2New = sol2[2].subs(x, xGrid[i])
    k2New = k2New.evalf(7)
    # добавляем координаты в пределах [0, 10] x [0, 10]
    if (k1New >= 0) and (k1New <= 10) and (k2New >= 0) and (k2New <= 10):
        k1CoordN.append(k1New)
        k2CoordN.append(k2New)

ax.clear()
ax.plot(k2CoordK, k1CoordK, label='Линия кратности')
ax.plot(k2CoordN, k1CoordN, label='Линия нейтральности', linestyle='--')
ax.set(xlabel='k2', ylabel='k1', title='k1&k2')
ax.legend(loc='upper center', fontsize='x-large')
fig.savefig(folder_name + "/k1&k2.png")

print("Двухпараметрический анализ по параметрам k1, k2 окончен")


#####################################################################################################################
# Построение фазовых портретов
print("Построение фазовых портретов начато ...")


# описываем функцию для решателя ОДУ
def model(f, t, p):
    x1, y1 = f
    k1, k1r, k2, k3, k3r = p
    z1 = 1 - x1 - y1
    system = [k1 * z1 - k1r * x1 - k2 * z1 ** 2 * x1,
              k3 * z1 ** 2 - k3r * y1]
    return system


# задаем значения параметров
k1 = 0.2
k2 = 6
k1r = k1rValueList[2]
k3 = k3Value
k3r = k3rValueList[1]

# задаем параметры решателя ОДУ
abserr = 1.0e-3
relerr = 1.0e-3
stoptime = 200.0
numpoints = 1000
t = np.linspace(0, 10000, numpoints)
p = [k1, k1r, k2, k3, k3r]

# начальные приближения
x0 = 0.842081571373749
y0 = 0.030303030303030304
w0 = [x0, y0]

# решаем систему
odeSolution = odeint(model, w0, t, args=(p,),
              atol=abserr, rtol=relerr)

# извлекаем координаты
xCoord = []
yCoord = []
for val in odeSolution:
    xCoord.append(val[0])
    yCoord.append(val[1])

ax.clear()
ax.plot(xCoord, yCoord)
portraitNumber = 11
title = str(portraitNumber) + ':_k1=' + str(k1) + '_k2=' + str(k2) + '_x0=' + str("{:.4f}".format(x0)) + \
        '_y0=' + str("{:.4f}".format(y0))
ax.set(xlabel='x', ylabel='y', title=title)
plt.show()

fig.savefig(folder_name + "/phase_portrait" + str(portraitNumber) + ".png")

print("Построение фазовых портретов окончено")