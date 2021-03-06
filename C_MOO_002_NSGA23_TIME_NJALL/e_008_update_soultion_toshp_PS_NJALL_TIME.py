# -*- coding:utf-8 -*-

"""
用于初始化provider对象
"""

import math
import json
#ORACLE库
import cx_Oracle
from pymongo import MongoClient
import a_mongo_operater
import numpy as np
from matplotlib.ticker import MultipleLocator


def update_solution_to_shp_layer_8_solutions(objectives_fitness,population_front0,sample_IDs,layer_name):
    # 删除重复行
    uniques = np.unique(objectives_fitness, axis=0)
    objectives_fitness_uniques = uniques[np.argsort(uniques[:, 2]), :]  # 按照第三列排序，自小往大排序，这样序号越小，建设成本越低，序号越大建设成本越高
    sample=objectives_fitness_uniques[sample_IDs,:]  # 获取到对应列的值
    print(sample)
    # 找出solution
    sample_solutions =[]
    for item in range(len(sample_IDs)):
        for j in range(objectives_fitness.shape[0]):
            if objectives_fitness[j,0]==sample[item,0] and  objectives_fitness[j,1]==sample[item,1] and  objectives_fitness[j,2]==sample[item,2]:
                sample_solutions.append(j)
                break
    conn = cx_Oracle.connect('powerstation/powerstation@localhost/ORCL')
    cursor = conn.cursor()
    provider_id_list = ['parks1', 'parks2', 'parks3', 'parks5', 'parks14', 'parks16', 'parks17', 'parks20', 'parks22', 'parks29', 'parks30', 'parks31', 'parks40', 'parks44', 'parks58', 'parks67', 'parks69', 'parks70', 'parks71', 'parks72', 'parks76', 'parks78', 'parks79', 'parks82', 'parks83', 'parks84', 'parks85', 'parks86', 'parks88', 'parks89', 'parks91', 'parks92', 'parks93', 'parks97', 'parks98', 'parks101', 'parks105', 'parks106', 'parks107', 'parks108', 'parks109', 'parks110', 'parks114', 'parks115', 'parks116', 'parks117', 'parks123', 'parks125', 'parks126', 'parks132', 'parks134', 'parks137', 'parks138', 'parks140', 'parks142', 'parks143', 'parks145', 'parks147', 'parks158', 'parks159', 'parks160', 'parks161', 'parks162', 'parks164', 'parks168', 'parks169', 'parks170', 'parks171', 'parks172', 'parks175', 'parks177', 'parks178', 'parks179', 'parks184', 'parks187', 'parks191', 'parks192', 'parks197', 'parks198', 'parks200', 'parks201', 'parks202', 'parks203', 'parks204', 'parks205', 'parks206', 'parks207', 'parks208', 'parks211', 'parks212', 'parks213', 'parks215', 'parks216', 'parks218', 'parks219', 'parks221', 'parks223', 'parks225', 'parks228', 'parks229', 'parks230', 'parks231', 'parks232', 'parks233', 'parks238', 'parks239', 'parks240', 'parks241', 'parks242', 'parks243', 'parks244', 'parks245', 'parks246', 'parks248', 'parks250', 'parks251', 'parks253', 'parks255', 'parks256', 'parks258', 'parks259', 'parks261', 'parks264', 'parks265', 'parks266', 'parks267', 'parks268', 'parks269', 'parks270', 'parks272', 'parks273', 'parks274', 'parks278', 'parks283', 'parks289', 'parks290', 'parks294', 'parks295', 'parks296', 'parks297', 'parks298', 'parks299', 'parks300', 'parks301', 'parks303', 'parks304', 'parks305', 'parks306', 'parks307', 'parks309', 'parks310', 'parks311', 'parks312', 'parks313', 'parks314', 'parks315', 'parks316', 'parks317', 'parks318', 'parks319', 'parks320', 'parks321', 'parks322', 'parks324', 'parks325', 'parks326', 'parks328', 'parks329', 'parks330', 'parks331', 'parks332', 'parks333', 'parks334', 'parks335', 'parks336', 'parks338', 'parks339', 'parks340', 'parks341', 'parks342', 'parks343', 'parks344', 'parks346', 'parks347', 'parks349', 'parks350', 'parks351', 'parks352', 'parks358', 'parks359', 'parks361', 'parks362', 'parks363', 'parks364', 'parks365', 'parks366', 'parks367', 'parks368', 'parks371', 'parks373', 'parks377', 'parks378', 'parks379', 'parks380', 'parks381', 'parks384', 'parks387', 'parks388', 'parks390', 'parks391', 'parks394', 'parks395', 'parks396', 'parks397', 'parks398', 'parks399', 'parks400', 'parks401', 'parks402', 'parks403', 'parks404', 'parks405', 'parks406', 'parks407', 'parks408', 'parks410', 'parks411', 'parks412', 'parks413', 'parks414', 'parks415', 'parks416', 'parks417', 'parks418', 'parks419', 'parks420', 'parks421', 'parks422', 'parks423', 'parks425', 'parks427', 'parks428', 'parks429', 'parks430', 'parks435', 'parks436', 'parks439', 'parks442', 'parks443', 'parks444', 'parks447', 'parks450', 'parks451', 'parks455', 'parks456', 'parks458', 'parks459', 'parks460', 'parks461', 'parks462', 'parks463', 'parks464', 'parks465', 'parks466', 'parks467', 'parks468', 'parks469', 'parks470', 'parks471', 'parks472', 'parks473', 'parks474', 'parks475', 'parks476', 'parks477', 'parks478', 'parks479', 'parks481', 'parks482', 'parks485', 'parks486', 'parks489', 'parks490', 'parks492', 'parks493', 'parks494', 'parks496', 'parks497', 'parks501', 'parks502', 'parks503', 'parks504', 'parks505', 'parks506', 'parks507', 'parks509', 'parks510', 'parks511', 'parks513', 'parks514', 'parks516', 'parks518', 'parks519', 'parks520', 'parks521', 'parks522', 'parks523', 'parks524', 'parks525', 'parks526', 'parks527', 'parks528', 'parks529', 'parks531', 'parks532', 'parks533', 'parks534', 'parks535', 'parks536', 'parks537', 'parks538', 'parks539', 'parks542', 'parks543', 'parks544', 'parks546', 'parks547', 'parks548', 'parks549', 'parks550', 'parks553', 'parks554', 'parks555', 'parks556', 'parks558', 'parks559', 'parks560', 'parks561', 'parks562', 'parks563', 'parks564', 'parks565', 'parks566', 'parks567', 'parks568', 'parks570', 'parks571', 'parks572', 'parks573', 'parks574', 'parks575', 'parks576', 'parks577', 'parks578', 'parks580', 'parks583', 'parks584', 'parks585', 'parks586', 'parks587', 'parks588', 'parks589', 'parks590', 'parks592', 'parks593', 'parks594', 'parks598', 'parks600', 'parks605', 'parks607', 'parks608', 'parks609', 'parks610', 'parks612', 'parks622', 'parks624', 'parks625', 'parks626', 'parks627', 'parks628', 'parks630', 'parks631', 'parks632', 'parks634', 'parks635', 'parks637', 'parks640', 'parks641', 'parks642', 'parks643', 'parks645', 'parks647', 'parks648', 'parks649', 'parks650', 'parks651', 'parks652', 'parks654', 'parks655', 'parks657', 'parks658', 'parks659', 'parks660', 'parks662', 'parks663', 'parks664', 'parks665', 'parks666', 'parks667', 'parks668', 'parks669', 'parks670', 'parks671', 'parks672', 'parks673', 'parks674', 'parks675', 'parks676', 'parks677', 'parks678', 'parks680', 'parks682', 'parks683', 'parks684', 'parks685', 'parks686', 'parks687', 'parks688', 'parks689', 'parks690', 'parks691', 'parks692', 'parks693', 'parks694', 'parks695', 'parks696', 'parks697', 'parks698', 'parks699', 'parks701', 'parks702', 'parks704', 'parks705', 'parks706', 'parks707', 'parks708', 'parks709', 'parks710', 'parks711', 'parks712', 'parks713', 'parks714', 'parks715', 'parks716', 'parks717', 'parks718', 'parks719', 'parks720', 'parks721', 'parks722', 'parks723', 'parks724', 'parks725', 'parks726', 'parks728', 'parks730', 'parks731', 'parks732', 'parks733', 'parks734', 'parks735', 'parks736', 'parks737', 'parks738', 'parks739', 'parks740', 'parks741', 'parks742', 'parks743', 'parks744', 'parks745', 'parks752', 'parks755', 'parks756', 'parks759', 'parks761', 'parks762', 'parks767', 'parks768', 'parks770', 'parks771', 'parks772', 'parks775', 'parks777', 'parks778', 'parks779', 'parks780', 'parks781', 'parks783', 'parks785', 'parks787', 'parks790', 'parks793', 'parks794', 'parks795', 'parks800', 'parks801', 'parks802', 'parks803', 'parks806', 'parks807', 'parks808', 'parks809', 'parks810', 'parks811', 'parks812', 'parks813', 'parks814', 'parks815', 'parks816', 'parks817', 'parks818', 'parks819', 'parks820', 'parks821', 'parks822', 'parks823', 'parks824', 'parks825', 'parks826', 'parks827', 'parks828', 'parks829', 'parks831', 'parks832', 'parks835', 'parks836', 'parks837', 'parks838', 'parks839', 'parks840', 'parks841', 'parks842', 'parks843', 'parks844', 'parks845', 'parks846', 'parks847', 'parks850', 'parks851', 'parks853', 'parks854', 'parks855', 'parks856', 'parks857', 'parks858', 'parks859', 'parks860', 'parks861', 'parks862', 'parks863', 'parks864', 'parks865', 'parks866', 'parks867', 'parks868', 'parks869', 'parks871', 'parks872', 'parks873']
    for rowid in range(len(provider_id_list)):
        update_sql = "update {0} set solution1={1},solution2={2}," \
                     "solution3={3},solution4={4},solution5={5}," \
                     "solution6={6},solution7={7},solution8={8}  where keyid_parks=\'{9}\'".format(
            layer_name,
            population_front0[sample_solutions[0],rowid ],
            population_front0[sample_solutions[1],rowid ],
            population_front0[sample_solutions[2],rowid ],
            population_front0[sample_solutions[3],rowid ],
            population_front0[sample_solutions[4],rowid ],
            population_front0[sample_solutions[5],rowid ],
            population_front0[sample_solutions[6],rowid ],
            population_front0[sample_solutions[7],rowid ],
            provider_id_list[rowid])
        cursor.execute(update_sql)
        conn.commit()
        print("字段更新成功")
    conn.close()



def update_solution_to_shp_layer_3_solutions(objectives_fitness,population_front0,sample_IDs,layer_name):
    # 删除重复行
    uniques = np.unique(objectives_fitness, axis=0)
    objectives_fitness_uniques = uniques[np.argsort(uniques[:, 2]), :]  # 按照第三列排序，自小往大排序，这样序号越小，建设成本越低，序号越大建设成本越高
    sample=objectives_fitness_uniques[sample_IDs,:]  # 获取到对应列的值
    print(sample)
    # 找出solution
    sample_solutions =[]
    for item in range(len(sample_IDs)):
        for j in range(objectives_fitness.shape[0]):
            if objectives_fitness[j,0]==sample[item,0] and  objectives_fitness[j,1]==sample[item,1] and  objectives_fitness[j,2]==sample[item,2]:
                sample_solutions.append(j)
                break
    conn = cx_Oracle.connect('powerstation/powerstation@localhost/ORCL')
    cursor = conn.cursor()
    provider_id_list = ['parks1', 'parks2', 'parks3', 'parks5', 'parks14', 'parks16', 'parks17', 'parks20', 'parks22', 'parks29', 'parks30', 'parks31', 'parks40', 'parks44', 'parks58', 'parks67', 'parks69', 'parks70', 'parks71', 'parks72', 'parks76', 'parks78', 'parks79', 'parks82', 'parks83', 'parks84', 'parks85', 'parks86', 'parks88', 'parks89', 'parks91', 'parks92', 'parks93', 'parks97', 'parks98', 'parks101', 'parks105', 'parks106', 'parks107', 'parks108', 'parks109', 'parks110', 'parks114', 'parks115', 'parks116', 'parks117', 'parks123', 'parks125', 'parks126', 'parks132', 'parks134', 'parks137', 'parks138', 'parks140', 'parks142', 'parks143', 'parks145', 'parks147', 'parks158', 'parks159', 'parks160', 'parks161', 'parks162', 'parks164', 'parks168', 'parks169', 'parks170', 'parks171', 'parks172', 'parks175', 'parks177', 'parks178', 'parks179', 'parks184', 'parks187', 'parks191', 'parks192', 'parks197', 'parks198', 'parks200', 'parks201', 'parks202', 'parks203', 'parks204', 'parks205', 'parks206', 'parks207', 'parks208', 'parks211', 'parks212', 'parks213', 'parks215', 'parks216', 'parks218', 'parks219', 'parks221', 'parks223', 'parks225', 'parks228', 'parks229', 'parks230', 'parks231', 'parks232', 'parks233', 'parks238', 'parks239', 'parks240', 'parks241', 'parks242', 'parks243', 'parks244', 'parks245', 'parks246', 'parks248', 'parks250', 'parks251', 'parks253', 'parks255', 'parks256', 'parks258', 'parks259', 'parks261', 'parks264', 'parks265', 'parks266', 'parks267', 'parks268', 'parks269', 'parks270', 'parks272', 'parks273', 'parks274', 'parks278', 'parks283', 'parks289', 'parks290', 'parks294', 'parks295', 'parks296', 'parks297', 'parks298', 'parks299', 'parks300', 'parks301', 'parks303', 'parks304', 'parks305', 'parks306', 'parks307', 'parks309', 'parks310', 'parks311', 'parks312', 'parks313', 'parks314', 'parks315', 'parks316', 'parks317', 'parks318', 'parks319', 'parks320', 'parks321', 'parks322', 'parks324', 'parks325', 'parks326', 'parks328', 'parks329', 'parks330', 'parks331', 'parks332', 'parks333', 'parks334', 'parks335', 'parks336', 'parks338', 'parks339', 'parks340', 'parks341', 'parks342', 'parks343', 'parks344', 'parks346', 'parks347', 'parks349', 'parks350', 'parks351', 'parks352', 'parks358', 'parks359', 'parks361', 'parks362', 'parks363', 'parks364', 'parks365', 'parks366', 'parks367', 'parks368', 'parks371', 'parks373', 'parks377', 'parks378', 'parks379', 'parks380', 'parks381', 'parks384', 'parks387', 'parks388', 'parks390', 'parks391', 'parks394', 'parks395', 'parks396', 'parks397', 'parks398', 'parks399', 'parks400', 'parks401', 'parks402', 'parks403', 'parks404', 'parks405', 'parks406', 'parks407', 'parks408', 'parks410', 'parks411', 'parks412', 'parks413', 'parks414', 'parks415', 'parks416', 'parks417', 'parks418', 'parks419', 'parks420', 'parks421', 'parks422', 'parks423', 'parks425', 'parks427', 'parks428', 'parks429', 'parks430', 'parks435', 'parks436', 'parks439', 'parks442', 'parks443', 'parks444', 'parks447', 'parks450', 'parks451', 'parks455', 'parks456', 'parks458', 'parks459', 'parks460', 'parks461', 'parks462', 'parks463', 'parks464', 'parks465', 'parks466', 'parks467', 'parks468', 'parks469', 'parks470', 'parks471', 'parks472', 'parks473', 'parks474', 'parks475', 'parks476', 'parks477', 'parks478', 'parks479', 'parks481', 'parks482', 'parks485', 'parks486', 'parks489', 'parks490', 'parks492', 'parks493', 'parks494', 'parks496', 'parks497', 'parks501', 'parks502', 'parks503', 'parks504', 'parks505', 'parks506', 'parks507', 'parks509', 'parks510', 'parks511', 'parks513', 'parks514', 'parks516', 'parks518', 'parks519', 'parks520', 'parks521', 'parks522', 'parks523', 'parks524', 'parks525', 'parks526', 'parks527', 'parks528', 'parks529', 'parks531', 'parks532', 'parks533', 'parks534', 'parks535', 'parks536', 'parks537', 'parks538', 'parks539', 'parks542', 'parks543', 'parks544', 'parks546', 'parks547', 'parks548', 'parks549', 'parks550', 'parks553', 'parks554', 'parks555', 'parks556', 'parks558', 'parks559', 'parks560', 'parks561', 'parks562', 'parks563', 'parks564', 'parks565', 'parks566', 'parks567', 'parks568', 'parks570', 'parks571', 'parks572', 'parks573', 'parks574', 'parks575', 'parks576', 'parks577', 'parks578', 'parks580', 'parks583', 'parks584', 'parks585', 'parks586', 'parks587', 'parks588', 'parks589', 'parks590', 'parks592', 'parks593', 'parks594', 'parks598', 'parks600', 'parks605', 'parks607', 'parks608', 'parks609', 'parks610', 'parks612', 'parks622', 'parks624', 'parks625', 'parks626', 'parks627', 'parks628', 'parks630', 'parks631', 'parks632', 'parks634', 'parks635', 'parks637', 'parks640', 'parks641', 'parks642', 'parks643', 'parks645', 'parks647', 'parks648', 'parks649', 'parks650', 'parks651', 'parks652', 'parks654', 'parks655', 'parks657', 'parks658', 'parks659', 'parks660', 'parks662', 'parks663', 'parks664', 'parks665', 'parks666', 'parks667', 'parks668', 'parks669', 'parks670', 'parks671', 'parks672', 'parks673', 'parks674', 'parks675', 'parks676', 'parks677', 'parks678', 'parks680', 'parks682', 'parks683', 'parks684', 'parks685', 'parks686', 'parks687', 'parks688', 'parks689', 'parks690', 'parks691', 'parks692', 'parks693', 'parks694', 'parks695', 'parks696', 'parks697', 'parks698', 'parks699', 'parks701', 'parks702', 'parks704', 'parks705', 'parks706', 'parks707', 'parks708', 'parks709', 'parks710', 'parks711', 'parks712', 'parks713', 'parks714', 'parks715', 'parks716', 'parks717', 'parks718', 'parks719', 'parks720', 'parks721', 'parks722', 'parks723', 'parks724', 'parks725', 'parks726', 'parks728', 'parks730', 'parks731', 'parks732', 'parks733', 'parks734', 'parks735', 'parks736', 'parks737', 'parks738', 'parks739', 'parks740', 'parks741', 'parks742', 'parks743', 'parks744', 'parks745', 'parks752', 'parks755', 'parks756', 'parks759', 'parks761', 'parks762', 'parks767', 'parks768', 'parks770', 'parks771', 'parks772', 'parks775', 'parks777', 'parks778', 'parks779', 'parks780', 'parks781', 'parks783', 'parks785', 'parks787', 'parks790', 'parks793', 'parks794', 'parks795', 'parks800', 'parks801', 'parks802', 'parks803', 'parks806', 'parks807', 'parks808', 'parks809', 'parks810', 'parks811', 'parks812', 'parks813', 'parks814', 'parks815', 'parks816', 'parks817', 'parks818', 'parks819', 'parks820', 'parks821', 'parks822', 'parks823', 'parks824', 'parks825', 'parks826', 'parks827', 'parks828', 'parks829', 'parks831', 'parks832', 'parks835', 'parks836', 'parks837', 'parks838', 'parks839', 'parks840', 'parks841', 'parks842', 'parks843', 'parks844', 'parks845', 'parks846', 'parks847', 'parks850', 'parks851', 'parks853', 'parks854', 'parks855', 'parks856', 'parks857', 'parks858', 'parks859', 'parks860', 'parks861', 'parks862', 'parks863', 'parks864', 'parks865', 'parks866', 'parks867', 'parks868', 'parks869', 'parks871', 'parks872', 'parks873']
    for rowid in range(len(provider_id_list)):
        update_sql = "update {0} set solution1={1},solution2={2},solution3={3}  where keyid_parks=\'{4}\'".format(
            layer_name,
            population_front0[sample_solutions[0],rowid ],
            population_front0[sample_solutions[1],rowid ],
            population_front0[sample_solutions[2],rowid ],
            provider_id_list[rowid])
        cursor.execute(update_sql)
        conn.commit()
        print("字段更新成功")
    conn.close()

def update_solution_to_shp_layer(objectives_fitness,population_front0,sample_IDs,layer_name):
    # 删除重复行
    uniques = np.unique(objectives_fitness, axis=0)
    objectives_fitness_uniques = uniques[np.argsort(uniques[:, 2]), :]  # 按照第三列排序，自小往大排序，这样序号越小，建设成本越低，序号越大建设成本越高
    sample=objectives_fitness_uniques[sample_IDs,:]  # 获取到对应列的值
    print(sample)
    # 找出solution
    sample_solutions =[]
    for item in range(len(sample_IDs)):
        for j in range(objectives_fitness.shape[0]):
            if objectives_fitness[j,0]==sample[item,0] and  objectives_fitness[j,1]==sample[item,1] and  objectives_fitness[j,2]==sample[item,2]:
                sample_solutions.append(j)
                break
    conn = cx_Oracle.connect('powerstation/powerstation@localhost/ORCL')
    cursor = conn.cursor()
    provider_id_list = ['parks1', 'parks2', 'parks3', 'parks5', 'parks14', 'parks16', 'parks17', 'parks20', 'parks22', 'parks29', 'parks30', 'parks31', 'parks40', 'parks44', 'parks58', 'parks67', 'parks69', 'parks70', 'parks71', 'parks72', 'parks76', 'parks78', 'parks79', 'parks82', 'parks83', 'parks84', 'parks85', 'parks86', 'parks88', 'parks89', 'parks91', 'parks92', 'parks93', 'parks97', 'parks98', 'parks101', 'parks105', 'parks106', 'parks107', 'parks108', 'parks109', 'parks110', 'parks114', 'parks115', 'parks116', 'parks117', 'parks123', 'parks125', 'parks126', 'parks132', 'parks134', 'parks137', 'parks138', 'parks140', 'parks142', 'parks143', 'parks145', 'parks147', 'parks158', 'parks159', 'parks160', 'parks161', 'parks162', 'parks164', 'parks168', 'parks169', 'parks170', 'parks171', 'parks172', 'parks175', 'parks177', 'parks178', 'parks179', 'parks184', 'parks187', 'parks191', 'parks192', 'parks197', 'parks198', 'parks200', 'parks201', 'parks202', 'parks203', 'parks204', 'parks205', 'parks206', 'parks207', 'parks208', 'parks211', 'parks212', 'parks213', 'parks215', 'parks216', 'parks218', 'parks219', 'parks221', 'parks223', 'parks225', 'parks228', 'parks229', 'parks230', 'parks231', 'parks232', 'parks233', 'parks238', 'parks239', 'parks240', 'parks241', 'parks242', 'parks243', 'parks244', 'parks245', 'parks246', 'parks248', 'parks250', 'parks251', 'parks253', 'parks255', 'parks256', 'parks258', 'parks259', 'parks261', 'parks264', 'parks265', 'parks266', 'parks267', 'parks268', 'parks269', 'parks270', 'parks272', 'parks273', 'parks274', 'parks278', 'parks283', 'parks289', 'parks290', 'parks294', 'parks295', 'parks296', 'parks297', 'parks298', 'parks299', 'parks300', 'parks301', 'parks303', 'parks304', 'parks305', 'parks306', 'parks307', 'parks309', 'parks310', 'parks311', 'parks312', 'parks313', 'parks314', 'parks315', 'parks316', 'parks317', 'parks318', 'parks319', 'parks320', 'parks321', 'parks322', 'parks324', 'parks325', 'parks326', 'parks328', 'parks329', 'parks330', 'parks331', 'parks332', 'parks333', 'parks334', 'parks335', 'parks336', 'parks338', 'parks339', 'parks340', 'parks341', 'parks342', 'parks343', 'parks344', 'parks346', 'parks347', 'parks349', 'parks350', 'parks351', 'parks352', 'parks358', 'parks359', 'parks361', 'parks362', 'parks363', 'parks364', 'parks365', 'parks366', 'parks367', 'parks368', 'parks371', 'parks373', 'parks377', 'parks378', 'parks379', 'parks380', 'parks381', 'parks384', 'parks387', 'parks388', 'parks390', 'parks391', 'parks394', 'parks395', 'parks396', 'parks397', 'parks398', 'parks399', 'parks400', 'parks401', 'parks402', 'parks403', 'parks404', 'parks405', 'parks406', 'parks407', 'parks408', 'parks410', 'parks411', 'parks412', 'parks413', 'parks414', 'parks415', 'parks416', 'parks417', 'parks418', 'parks419', 'parks420', 'parks421', 'parks422', 'parks423', 'parks425', 'parks427', 'parks428', 'parks429', 'parks430', 'parks435', 'parks436', 'parks439', 'parks442', 'parks443', 'parks444', 'parks447', 'parks450', 'parks451', 'parks455', 'parks456', 'parks458', 'parks459', 'parks460', 'parks461', 'parks462', 'parks463', 'parks464', 'parks465', 'parks466', 'parks467', 'parks468', 'parks469', 'parks470', 'parks471', 'parks472', 'parks473', 'parks474', 'parks475', 'parks476', 'parks477', 'parks478', 'parks479', 'parks481', 'parks482', 'parks485', 'parks486', 'parks489', 'parks490', 'parks492', 'parks493', 'parks494', 'parks496', 'parks497', 'parks501', 'parks502', 'parks503', 'parks504', 'parks505', 'parks506', 'parks507', 'parks509', 'parks510', 'parks511', 'parks513', 'parks514', 'parks516', 'parks518', 'parks519', 'parks520', 'parks521', 'parks522', 'parks523', 'parks524', 'parks525', 'parks526', 'parks527', 'parks528', 'parks529', 'parks531', 'parks532', 'parks533', 'parks534', 'parks535', 'parks536', 'parks537', 'parks538', 'parks539', 'parks542', 'parks543', 'parks544', 'parks546', 'parks547', 'parks548', 'parks549', 'parks550', 'parks553', 'parks554', 'parks555', 'parks556', 'parks558', 'parks559', 'parks560', 'parks561', 'parks562', 'parks563', 'parks564', 'parks565', 'parks566', 'parks567', 'parks568', 'parks570', 'parks571', 'parks572', 'parks573', 'parks574', 'parks575', 'parks576', 'parks577', 'parks578', 'parks580', 'parks583', 'parks584', 'parks585', 'parks586', 'parks587', 'parks588', 'parks589', 'parks590', 'parks592', 'parks593', 'parks594', 'parks598', 'parks600', 'parks605', 'parks607', 'parks608', 'parks609', 'parks610', 'parks612', 'parks622', 'parks624', 'parks625', 'parks626', 'parks627', 'parks628', 'parks630', 'parks631', 'parks632', 'parks634', 'parks635', 'parks637', 'parks640', 'parks641', 'parks642', 'parks643', 'parks645', 'parks647', 'parks648', 'parks649', 'parks650', 'parks651', 'parks652', 'parks654', 'parks655', 'parks657', 'parks658', 'parks659', 'parks660', 'parks662', 'parks663', 'parks664', 'parks665', 'parks666', 'parks667', 'parks668', 'parks669', 'parks670', 'parks671', 'parks672', 'parks673', 'parks674', 'parks675', 'parks676', 'parks677', 'parks678', 'parks680', 'parks682', 'parks683', 'parks684', 'parks685', 'parks686', 'parks687', 'parks688', 'parks689', 'parks690', 'parks691', 'parks692', 'parks693', 'parks694', 'parks695', 'parks696', 'parks697', 'parks698', 'parks699', 'parks701', 'parks702', 'parks704', 'parks705', 'parks706', 'parks707', 'parks708', 'parks709', 'parks710', 'parks711', 'parks712', 'parks713', 'parks714', 'parks715', 'parks716', 'parks717', 'parks718', 'parks719', 'parks720', 'parks721', 'parks722', 'parks723', 'parks724', 'parks725', 'parks726', 'parks728', 'parks730', 'parks731', 'parks732', 'parks733', 'parks734', 'parks735', 'parks736', 'parks737', 'parks738', 'parks739', 'parks740', 'parks741', 'parks742', 'parks743', 'parks744', 'parks745', 'parks752', 'parks755', 'parks756', 'parks759', 'parks761', 'parks762', 'parks767', 'parks768', 'parks770', 'parks771', 'parks772', 'parks775', 'parks777', 'parks778', 'parks779', 'parks780', 'parks781', 'parks783', 'parks785', 'parks787', 'parks790', 'parks793', 'parks794', 'parks795', 'parks800', 'parks801', 'parks802', 'parks803', 'parks806', 'parks807', 'parks808', 'parks809', 'parks810', 'parks811', 'parks812', 'parks813', 'parks814', 'parks815', 'parks816', 'parks817', 'parks818', 'parks819', 'parks820', 'parks821', 'parks822', 'parks823', 'parks824', 'parks825', 'parks826', 'parks827', 'parks828', 'parks829', 'parks831', 'parks832', 'parks835', 'parks836', 'parks837', 'parks838', 'parks839', 'parks840', 'parks841', 'parks842', 'parks843', 'parks844', 'parks845', 'parks846', 'parks847', 'parks850', 'parks851', 'parks853', 'parks854', 'parks855', 'parks856', 'parks857', 'parks858', 'parks859', 'parks860', 'parks861', 'parks862', 'parks863', 'parks864', 'parks865', 'parks866', 'parks867', 'parks868', 'parks869', 'parks871', 'parks872', 'parks873']
    for rowid in range(len(provider_id_list)):
        update_sql = "update {0} set solution1={1},solution2={2}  where keyid_parks=\'{3}\'".format(
            layer_name,
            population_front0[sample_solutions[0],rowid ],
            population_front0[sample_solutions[1],rowid ],
            provider_id_list[rowid])
        cursor.execute(update_sql)
        conn.commit()
        print("字段更新成功")
    conn.close()

    # for column in range(len(sample_solutions)):
    #     update_sql = "update popu0412_join_parks_join_ps_po set solution1={0},solution2={1},solution3={2},solution4={3},solution5={4},solution6={5},solution7={6},solution8={7}  where keyid_parks=\'{8}\'".format(
    #         population_front0[0, sample_solutions[column]], population_front0[1, sample_solutions[column]], population_front0[2, sample_solutions[column]], population_front0[3, sample_solutions[column]],
    #         population_front0[4, sample_solutions[column]], population_front0[5, sample_solutions[column]], population_front0[6, sample_solutions[column]],population_front0[7, sample_solutions[column]],
    #         provider_id_list[sample_solutions[column]])
    #     cursor.execute(update_sql)
    #     conn.commit()
    #     print("save solution {0}".format(sample_solutions[column]))
    # conn.close()

    # for column in range(population_front0.shape[1]):
    #     update_sql = "update popu0412_join_parks_join_ps_po set solution1={0},solution2={1},solution3={2},solution4={3},solution5={4},solution6={5},solution7={6},solution8={7},solution9={8},solution10={9}  where keyid_parks=\'{10}\'".format(
    #         population_front0[0, column], population_front0[1, column], population_front0[2, column], population_front0[3, column],
    #         population_front0[4, column], population_front0[5, column], population_front0[6, column],population_front0[7, column],
    #         population_front0[8, column], population_front0[9, column],provider_id_list[column])
    #     cursor.execute(update_sql)
    #     conn.commit()
    # conn.close()


if __name__=="__main__":
    # #方案一： drivingtime
    # log_file_path = "./log"
    # D1_npzfile = np.load(log_file_path + "/njall_solutions_nsga2_up200_time033_drivingtime.npz")
    # population_front0 = D1_npzfile["population_front0"]
    # objectives_fitness = D1_npzfile["objectives_fitness"]
    # sample_IDs=[0, 2, 19, 20, 33, 34, 59, 68]   # 人工选择的，根据之前的分布情况
    # layer_name = "popu0412_join_parks_join_ps_po"
    # update_solution_to_shp_layer_8_solutions(objectives_fitness,population_front0,sample_IDs,layer_name)

    # 方案二： coverpeople
    log_file_path = "./log"
    D1_npzfile = np.load(log_file_path + "/njall_solutions_nsga2_up200_time033_coverpeople.npz")
    population_front0 = D1_npzfile["population_front0"]
    objectives_fitness = D1_npzfile["objectives_fitness"]
    sample_IDs = [0, 53, 87]  # 人工选择的，根据之前的分布情况
    layer_name = "popu0412_coverpeople"
    update_solution_to_shp_layer_3_solutions(objectives_fitness, population_front0, sample_IDs, layer_name)


    #方案三：static
    # log_file_path = "./log"
    # D1_npzfile = np.load(log_file_path + "/njall_solutions_nsga2_up100_time0.16_poplimit_static.npz")
    # population_front0 = D1_npzfile["population_front0"]
    # objectives_fitness = D1_npzfile["objectives_fitness"]
    # sample_IDs = [0, 54]  # 人工选择的，根据之前的分布情况
    # layer_name="popu0412_static"
    #
    # # [[0.58445644 0.7851305  0.]
    # #  [0.13062353 0.29993257 1.]]
    # # save
    # # solution
    # # 70
    # # save
    # # solution
    # # 158
    # print(population_front0[70,:])
    # print(population_front0[158,:])
    #
    # # # #dynamic
    # log_file_path = "./log"
    # D1_npzfile = np.load(log_file_path + "/njall_solutions_nsga2_up100_time0.16_poplimit_dynamic.npz")
    # population_front0 = D1_npzfile["population_front0"]
    # objectives_fitness = D1_npzfile["objectives_fitness"]
    # sample_IDs = [0, 98]  # 人工选择的，根据之前的分布情况
    # layer_name="popu0412_dynamic"
    #
    # # [[0.451003   0.74322313 0.]
    # #  [0.16933286 0.05132197 1.]]
    # # save
    # # solution
    # # 125
    # # save
    # # solution
    # # 142
    # print(population_front0[125,:])
    # print(population_front0[142,:])
    # update_solution_to_shp_layer(objectives_fitness,population_front0,sample_IDs,layer_name)
