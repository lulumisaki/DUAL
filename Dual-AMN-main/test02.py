# import numpy as np
#
# # 创建一个示例二维数组
# array = np.array([[1, 5, 3, 9, 10],
#                   [4, 2, 8, 7, 6],
#                   [12, 11, 13, 15, 14]])
# # 找到每行的第十大值的索引
# indices = np.argpartition(array, -3, axis=1)[:, -3:]
# # 使用索引获取每行的第十大值及其之后的值
# new_array = np.array([row[row >= np.partition(row, -3)[-3]] for row in array])
# print(new_array)
import numpy as np

# 创建一个示例二维数组
array = np.array([[3, 1, 4],
                  [5, 2, 9],
                  [7, 6, 8]])

# 对每行的值进行排序（从大到小）
sorted_array = np.sort(array, axis=1)[:, ::-1]
idx = 0
for item in sorted_array:
    print(item)
