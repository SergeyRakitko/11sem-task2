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

# jacACount1, jacACount2 - промежуточные подсчеты Якобиана в числовом виде
jacACount1 = jacA.subs({k1: k1Value, k3: k3Value, k3r: k3rValueList[1]})
meshSize = 100  # плотность сетки по переменной 'y'
yGrid = np.linspace(0, 0.5, meshSize, dtype=float)

# для всех значений параметра k1r
for j in range(2):  # len(k1rValueList)):
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
                jacACount2 = jacACount1.subs({y: yNew, x: xNew, k2: k2New, k1r: k1rValueList[j]})
                jacACount2 = jacACount2.evalf(7)
                # ищем собственные значения
                eigenValues = jacACount2.eigenvals()
                negativeValues = True
                zeroValue = False
                bothZeroValues = True
                for key, val in eigenValues.items():
                    # вещественная часть
                    realPart = (complex(key)).real
                    # оба отрицательные
                    negativeValues = negativeValues and (realPart < 0)
                    # есть хотя бы одно нулевое
                    zeroValue = zeroValue or (math.fabs(realPart) < 0.0051)
                    # оба нулевые
                    bothZeroValues = bothZeroValues and (math.fabs(realPart) < 0.0051)
                if bothZeroValues:
                    # бифуркация Андронова-Хопфа
                    horBif2List.append(k2New)
                    verBif2List.append(yNew)
                elif negativeValues and zeroValue:
                    # седло-узловая бифуркация
                    horBif1List.append(k2New)
                    verBif1List.append(yNew)
    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='Седло-узловые')
    ax.plot(horBif2List, verBif2List, 'ro', label='Андронова-Хопфа', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k1r=' + str(k1rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')
    plt.show()

    # сохраняем
    fig.savefig(folder_name + "/k1r=" + str(k1rValueList[j]) + ".png")

print("Однопараметрический анализ по параметру k1r окончен")

##########################################################
# однопараметрический анализ по параметру k3r
print("Однопараметрический анализ по параметру k3r начат ...")

# выбираем решение, подставляем известные параметры
solNumber = 0  # 0, 1, второе решение не лежит в ОДЗ
sol = [staticSolution[solNumber][0].subs({k1: k1Value, k3: k3Value, k1r: k1rValueList[2]}),
       staticSolution[solNumber][1].subs({k1: k1Value, k3: k3Value, k1r: k1rValueList[2]})]

# jacACount1, jacACount2 - промежуточные подсчеты Якобиана в числовом виде
jacACount1 = jacA.subs({k1: k1Value, k3: k3Value, k1r: k1rValueList[2]})
meshSize = 100  # плотность сетки по переменной 'y'
yGrid = np.linspace(0, 0.5, meshSize, dtype=float)

# для всех значений параметра k3r
for j in range(2):  # len(k3rValueList)):
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
                jacACount2 = jacACount1.subs({y: yNew, x: xNew, k2: k2New, k3r: k3rValueList[j]})
                jacACount2 = jacACount2.evalf(7)
                # ищем собственные значения
                eigenValues = jacACount2.eigenvals()
                negativeValues = True
                zeroValue = False
                bothZeroValues = True
                for key, val in eigenValues.items():
                    # вещественная часть
                    realPart = (complex(key)).real
                    # оба отрицательные
                    negativeValues = negativeValues and (realPart < 0)
                    # есть хотя бы одно нулевое
                    zeroValue = zeroValue or (math.fabs(realPart) < 0.00051)
                    # оба нулевые
                    bothZeroValues = bothZeroValues and (math.fabs(realPart) < 0.00051)
                if bothZeroValues:
                    # бифуркация Андронова-Хопфа
                    horBif2List.append(k2New)
                    verBif2List.append(yNew)
                elif negativeValues and zeroValue:
                    # седло-узловая бифуркация
                    horBif1List.append(k2New)
                    verBif1List.append(yNew)
    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='Седло-узловые')
    ax.plot(horBif2List, verBif2List, 'ro', label='Андронова-Хопфа', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k3r=' + str(k3rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')
    plt.show()

    # сохраняем
    fig.savefig(folder_name + "/k3r=" + str(k3rValueList[j]) + ".png")

print("Однопараметрический анализ по параметру k3r окончен")


####################################################################################################################
# двухпараметрический анализ по параметрам k1, k2
print("Двухпараметрический анализ по параметрам k1, k2 начат ...")

# подставляем известные параметры
f1Count1 = f1.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
f2Count1 = f2.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
determinantCount1 = determinant.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
# решаем расширенную систему
complexSolution = solve([f1Count1, f2Count1, determinantCount1], y, k1, k2)
sol = complexSolution[0]
# сетка по переменной 'x'
xGrid = np.linspace(0, 0.9873, 100, dtype=float)
k1Coord = []
k2Coord = []
for i in range(len(xGrid)):
    # вычисляем значение k1
    k1New = sol[1].subs(x, xGrid[i])
    k1New = k1New.evalf(7)
    # вычисляем значение k2
    k2New = sol[2].subs(x, xGrid[i])
    k2New = k2New.evalf(7)
    # добавляем координаты в пределах [0, 10] x [0, 10]
    if (k1New >= 0) and (k1New <= 10) and (k2New >= 0) and (k2New <= 10):
        k1Coord.append(k1New)
        k2Coord.append(k2New)

ax.clear()
ax.plot(k2Coord, k1Coord)
ax.set(xlabel='k2', ylabel='k1', title='k1&k2')
plt.show()

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
k1 = 0.5
k2 = 2
k1r = k1rValueList[2]
k3 = k3Value
k3r = k3rValueList[1]

# задаем параметры решателя ОДУ
abserr = 1.0e-3
relerr = 1.0e-3
stoptime = 50.0
numpoints = 100
t = np.linspace(0, 50, numpoints)
p = [k1, k1r, k2, k3, k3r]

# начальные приближения
x0 = 0.940070371912747
y0 = 0.510101010101010102
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
ax.plot(t, yCoord, label='y')
ax.plot(t, xCoord, label='x')
ax.set(xlabel='x', ylabel='y', title='Phase portrait')
ax.legend(loc='upper center', fontsize='x-large')
plt.show()

fig.savefig(folder_name + "/Phase_portrait.png")

print("Построение фазовых портретов окончено")
