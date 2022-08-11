# -*- coding: utf-8 -*-
'''
1. Histogram - statistik
2. Bar chart
3. Pie chart
4. Line chart
5. Scatter plot


'''

# Line graph 
age = [5,10,15,20,25,30,35,40]
height = [80, 100, 150, 165, 170, 170, 170, 170 ] 

import matplotlib.pyplot as plt
# from pyplot import matplotlib as plt

plt.plot(age, height, linewidth= 4, linestyle='dashed')
plt.title("Age and Height Distribution")
plt.xlabel("Age")
plt.ylabel("Height")
# plt.show()

# Scatter plot
plt.scatter(age, height)

# Pie Chart
course = ["UX", "DS", "Web", "C", "Java"]
say = [30, 20, 50, 10, 100]

plt.pie(say, explode = (0,0,0,0,0.1), colors = ('#06212A', '#ABF0CF', '#DAABF0', '#F0C3AB', '#F0ABC3' ) , labels = course, shadow= True)



# Bar Chart
plt.bar(course, say)

# Seaborn 
'''
Relational Plot
Regresiya plotu
Distributional Plot
Categorical Plot

'''

import seaborn as sns

sns.set_theme()

tip = sns.load_dataset('tips')
sns.relplot(data = tip, x= 'total_bill', y = 'tip', col = 'sex', hue= 'smoker', style= 'day' , size= 'size')   #palette?

# style, size
sns.catplot(data = tip, x= 'day', y = 'tip', col = 'time', hue = 'sex', size= 'size')

# distributional plot
sns.displot(data = tip, x= 'total_bill', col = 'time', hue = 'sex', kde = True)


