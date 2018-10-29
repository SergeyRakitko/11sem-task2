from sympy import symbols, expand, factor, diff, Matrix, solve
import numpy as np
import math
from scipy import linalg
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# значения параметров модели
k1Value = 0.12
k1rValueList = [0.001, 0.005, 0.01, 0.015, 0.02]
k2Value = 2.5
k3Value = 0.0032
k3rValueList = [0.0005, 0.001, 0.002, 0.003, 0.004]

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

# подставляем фиксированные параметры и решаем систему
#f1 = f1.subs({k1: k1Value, k3: k3Value})
#f2 = f2.subs({k1: k1Value, k3: k3Value})
#jacA = jacA.subs({k1: k1Value, k3: k3Value})
# determinant = determinant.subs({k1: k1Value, k3: k3Value})
#trace = trace.subs({k1: k1Value, k3: k3Value})

#res = solve([f1, f2], x, k2)
"""
N = 100  # плотность сетки
# однопараметрический анализ по параметру k1r
solutionNumber = 0  # 0, 1
sol1 = [res[solutionNumber][0].subs(k3r, k3rValueList[1]), res[solutionNumber][1].subs(k3r, k3rValueList[1])]
jacA1 = jacA.subs(k3r, k3rValueList[1])
determinant1 = determinant.subs(k3r, k3rValueList[1])
trace1 = trace.subs(k3r, k3rValueList[1])
fig, ax = plt.subplots()

# для всех значений параметра k1r
for j in range(2):  # len(k1rValueList)):
    yGrid = np.linspace(0, 0.5, N, dtype=float)
    yCoord = []
    k2Coord = []
    xCoord = []
    # седло-узловая биффуркация
    horBif1List = []
    verBif1List = []
    # биффуркация Андронова-Хопфа
    horBif2List = []
    verBif2List = []
    for i in range(N):
        ynew = yGrid[i]
        k2new = sol1[1].subs({y: ynew, k1r: k1rValueList[j]})
        xnew = sol1[0].subs({y: ynew, k1r: k1rValueList[j]})

        if not (math.isnan(xnew) or math.isnan(ynew) or math.isnan(k2new)):
            if (xnew >= 0) and (xnew <= 1) and (xnew + 2 * ynew >= 0) and (xnew + 2 * ynew <= 1):
                yCoord.append(ynew)
                xCoord.append(xnew)
                k2Coord.append(k2new)
                jacANum = jacA1.subs({y: ynew, x: xnew, k2: k2new, k1r: k1rValueList[j]})
                determinantNum = determinant1.subs({y: ynew, x: xnew, k2: k2new, k1r: k1rValueList[j]})
                traceNum = trace1.subs({y: ynew, x: xnew, k2: k2new, k1r: k1rValueList[j]})
                jacANum = jacANum.evalf(7)
                determinantNum = determinantNum.evalf(7)
                traceNum = traceNum.evalf(7)
                vals = jacANum.eigenvals()
                negativeValues = True
                zeroValue = False
                bothZeroValues = True
                for key, val in vals.items():
                    realPart = (complex(key)).real
                    negativeValues = negativeValues and (realPart < 0)
                    zeroValue = zeroValue or (math.fabs(realPart) < 0.0051)
                    bothZeroValues = bothZeroValues and (math.fabs(realPart) < 0.0051)
                if bothZeroValues:
                    horBif2List.append(k2new)
                    verBif2List.append(ynew)
                elif negativeValues and zeroValue:
                    horBif1List.append(k2new)
                    verBif1List.append(ynew)

    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='bif1')
    ax.plot(horBif2List, verBif2List, 'ro', label='bif2', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k1r=' + str(k1rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')

    # сделать создание папки, если ее нет
    fig.savefig("results/k1r=" + str(k1rValueList[j]) + ".png")
    plt.show()
"""
#################################################
"""
N = 100  # плотность сетки
# однопараметрический анализ по параметру k3r
solutionNumber = 0  # 0, 1
sol1 = [res[solutionNumber][0].subs(k1r, k1rValueList[2]), res[solutionNumber][1].subs(k1r, k1rValueList[2])]
jacA1 = jacA.subs(k1r, k1rValueList[2])
determinant1 = determinant.subs(k1r, k1rValueList[2])
trace1 = trace.subs(k1r, k1rValueList[2])
fig, ax = plt.subplots()

# для всех значений параметра k3r
for j in range(len(k3rValueList)):
    yGrid = np.linspace(0, 0.5, N, dtype=float)
    yCoord = []
    k2Coord = []
    xCoord = []
    # седло-узловая биффуркация
    horBif1List = []
    verBif1List = []
    # биффуркация Андронова-Хопфа
    horBif2List = []
    verBif2List = []
    for i in range(N):
        ynew = yGrid[i]
        k2new = sol1[1].subs({y: ynew, k3r: k3rValueList[j]})
        xnew = sol1[0].subs({y: ynew, k3r: k3rValueList[j]})

        if not(math.isnan(xnew) or math.isnan(ynew) or math.isnan(k2new)):
            if (xnew >= 0)and(xnew <= 1)and(xnew + 2*ynew >= 0)and(xnew + 2*ynew <= 1):
                yCoord.append(ynew)
                xCoord.append(xnew)
                k2Coord.append(k2new)
                jacANum = jacA1.subs({y: ynew, x: xnew, k2: k2new, k3r: k3rValueList[j]})
                determinantNum = determinant1.subs({y: ynew, x: xnew, k2: k2new, k3r: k3rValueList[j]})
                traceNum = trace1.subs({y: ynew, x: xnew, k2: k2new, k3r: k3rValueList[j]})
                jacANum = jacANum.evalf(7)
                determinantNum = determinantNum.evalf(7)
                traceNum = traceNum.evalf(7)
                vals = jacANum.eigenvals()
                negativeValues = True
                zeroValue = False
                bothZeroValues = True
                for key, val in vals.items():
                    realPart = (complex(key)).real
                    negativeValues = negativeValues and (realPart < 0)
                    zeroValue = zeroValue or (math.fabs(realPart) < 0.00051)
                    bothZeroValues = bothZeroValues and (math.fabs(realPart) < 0.00051)
                if bothZeroValues:
                    horBif2List.append(k2new)
                    verBif2List.append(ynew)
                elif negativeValues and zeroValue:
                    horBif1List.append(k2new)
                    verBif1List.append(ynew)
                    #print(determinantNum)
                    #print(traceNum)
                    #for key, val in vals.items():
                    #    print(format((complex(key)).real, '.15f'), val)

    # изображаем
    ax.clear()
    ax.plot(k2Coord, yCoord, label='y')
    ax.plot(k2Coord, xCoord, label='x')
    ax.plot(horBif1List, verBif1List, 'ro', label='bif1')
    ax.plot(horBif2List, verBif2List, 'ro', label='bif2', color='green')
    ax.set(xlabel='k2', ylabel='x, y', title='k3r=' + str(k3rValueList[j]))
    ax.legend(loc='upper center', fontsize='x-large')

    # сделать создание папки, если ее нет
    fig.savefig("results/k3r=" + str(k3rValueList[j]) + ".png")
    plt.show()

"""

# двухпараметрический анализ
f1 = f1.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
f2 = f2.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
determinant = determinant.subs({k1r: k1rValueList[2], k3r: k3rValueList[1], k3: k3Value})
res2 = solve([f1, f2, determinant], y, k1, k2)
#res2 = res2.subs({y: ynew, x: xnew, k2: k2new, k3r: k3rValueList[j]})
for val in res2:
    for al in val:
        print(al)

xGrid = np.linspace(0, 0.9873, 100, dtype=float)
k1Coord = []
k2Coord = []
for i in range(len(xGrid)):
    val1 = res2[0][1].subs(x, xGrid[i])
    val1 = val1.evalf(7)
    val2 = res2[0][2].subs(x, xGrid[i])
    val2 = val2.evalf(7)
    if (val1 >= 0) and (val1 <= 10)and(val2 >= 0) and (val2 <= 10):
        k1Coord.append(val1)
        k2Coord.append(val2)


fig, ax = plt.subplots()
ax.plot(k2Coord, k1Coord)
ax.set(xlabel='k2', ylabel='k1', title='k1 & k2')
#fig.savefig("results/k1r=" + str(k1rValueList[j]) + ".png")
plt.show()