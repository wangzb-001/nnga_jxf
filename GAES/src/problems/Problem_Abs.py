# from abc import abstractmethod
#
#
# class Mymeta(type):
#     def __init__(self, class_name, class_bases, class_dic):
#         self.class_name = class_name
#         # 重用父类type的功能
#         super().__init__(class_name, class_bases,
#                          class_dic)
#
#
# class Problem_Abs(object, metaclass=Mymeta):
#
#     def __init__(self, valuable_num, Bound, pop_size, iterator_num, golbal_fv, accuracy):
#         self.valuable_num = valuable_num
#         self.Bound = Bound
#         self.pop_size = pop_size
#         self.iterator_num = iterator_num
#         self.golbal_fv = golbal_fv
#         self.accuracy = accuracy
#
#     def __str__(self):
#         print("class_name:", self.class_name)
#         return self.class_name
#
#     @abstractmethod
#     def Func(self, X):
#         pass
