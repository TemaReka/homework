import numpy as py
import matplotlib.pyplot as plt
import math as mh
import numpy as np


#Заданные заданием пределы. Шаг выбран произвольно
x_min = -512
x_max = 512
dx = 0.1
A = 512

#Создание массива точек с заданным шагом и их подстановка в функцию y
x = py.arange(x_min , x_max + 1, dx)
y = -(A+47)*py.sin(py.sqrt(py.abs((x/2)+(A+47)))) - x*py.sin(py.sqrt(py.abs(x-(A+47))))

#Подсчет количества точек
num_max = int(((abs(x_max)+abs(x_min))/dx)+1)
num = py.arange(1, num_max + 1)

#Заталкивание всех точек в один массив
n = 0
out = []
while n < num_max:
  out.append((num[n], x[n], y[n]))
  n += 1

#out += [[num[n]] + [x[n]] + [y[n]]]

  
# Сохраняем результаты в текстовый файл
with open(r"1/result/result.txt", "w") as file:
    for i in range(len(x)):
        file.write("{:.4f}    {:.4f}, \n".format(x[i], y[i]))

#Построение графика
plt.plot(x,y)   
plt.title("Line graph")   
plt.ylabel('Y axis')

plt.xlabel('X axis')
plt.xlim(x_min, x_max)

plt.show()

#Вывод итоговых значений
print(np.array(out))
print("\nПрограмма завершила свою работу")











































