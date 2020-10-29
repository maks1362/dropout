import sklearn
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sys import argv


def format_dataset(students: str, marks: str) -> pd.DataFrame:
    """
    Преобразовывает данные в датасете оцифровывает колонки
    :param students: адрес файла 'csv' со студентами
    :param marks: адрес файла 'csv' с оценками
    :return: DataFrame, таблица результата преобразований
    """

    import pandas as pd

    # read files
    data_stud = pd.read_csv(students, sep=';', error_bad_lines=False)
    data = pd.read_csv(marks, sep=';', error_bad_lines=False)

    # Удаление зачётов
    df_filter = data['ball'].isin([0])
    df_filter2 = data['namevidk'].isin(['зачет'])
    # data[df_filter2]
    data = data.query('namevidk not in ["зачет"]')
    # data[df_filter2]
    data = data.query('ball not in [0]')

    # Считаем среднее по оценкам
    # df_filter = data['namevidk'].isin(['1-я м/с аттестация','2-я м/с аттестация'])
    arr = data.groupby('proid')['ball'].mean()

    # Изменение "," на "."
    data_stud["att_srball"] = data_stud["att_srball"].str.replace(',', ".").astype(float)

    # Меняем 6 на 0 в колонке razdel
    data_stud['razdel'].replace([6], [0], inplace=True)
    # Заменяем пол на жен = 0, муж = 1
    data_stud['pol'].replace(['жен', 'муж'], [0, 1], inplace=True)
    # б=0, д = 1
    data_stud['dkb'].replace(['  б', 'д  '], [0, 1], inplace=True)
    # Убираем NaN в books
    data_stud['books'] = data_stud['books'].fillna(0)
    # Проживает = 1
    data_stud['Общежитие'].replace(['Проживает'], [1], inplace=True)
    # NaN = 0
    data_stud['Общежитие'] = data_stud['Общежитие'].fillna(0)
    # NaN = 0 в колоке kurs
    data_stud['kurs'] = data_stud['kurs'].fillna(0)
    # NaN = 0 в колоке группа
    data_stud['Группа'] = data_stud['Группа'].fillna(0)

    # Добавить "м" к магистратуре
    for i in range(data_stud.shape[0]):
        if data_stud['Группа'][i] != 0:
            if data_stud['Группа'][i][len(data_stud['Группа'][i]) - 1].lower() == 'м':
                data_stud['kurs'][i] = int(data_stud['Группа'][i][len(data_stud['Группа'][i]) - 3]) + 10
    for i in range(data_stud.shape[0]):
        data_stud['Группа'][i] = str(data_stud['Группа'][i]).split('-')[0]

    # Словарь для направлений\групп
    level_map = {0: 0, 'АД': 1, 'АНТМ': 2, 'АСД': 3, 'АСДМ': 4, 'АТ': 5, 'АЭМ': 6, 'АЭТМ': 7, 'МА': 8, 'МВ': 9,
                 'ММ': 10,
                 'МО': 11, 'МСФ': 12, 'МТ': 13, 'ОВР': 14, 'ПГС': 15, 'САР': 16, 'САРД': 17, 'ХТ': 18, 'ХТБ': 19,
                 'ХТБМ': 20,
                 'ХТК': 21, 'ХТЛ': 22, 'ХТЛМ': 23, 'ХТМ': 24, 'ХТО': 25, 'ХТОМ': 26, 'ХТОС': 27, 'ХТП': 28, 'ХТХ': 29,
                 'ХТХБ': 30, 'ХТХМ': 31, 'ХТЭ': 32, 'ЭИ': 33, 'ЭИС': 34, 'ЭМ': 35, 'ЭМИС': 36, 'ЭМЛ': 37, 'ЭММП': 38,
                 'ЭМП': 39, 'ЭМУК': 40, 'ЭМЭФ': 41, 'ЭПИ': 42, 'ЭСМ': 43, 'ЭУК': 44, 'ЭЭ': 45}
    data_stud['Группа'] = data_stud['Группа'].map(level_map)

    # Переименование на английский колонок
    data_stud.rename(columns={'Общежитие': 'obsh', 'Группа': 'group'}, inplace=True)

    # Удаляем столбцы точно не использующуеся в модели обучения
    data_stud.drop('Раздел', axis=1, inplace=True)
    data_stud.drop('Фак', axis=1, inplace=True)
    data_stud.drop('Numbcard', axis=1, inplace=True)
    data_stud.drop('data_okonch', axis=1, inplace=True)
    data_stud.drop('ia_ball', axis=1, inplace=True)
    data_stud.drop('ex_text', axis=1, inplace=True)
    data_stud.drop('ia_text', axis=1, inplace=True)
    data_stud.drop('addrleft', axis=1, inplace=True)
    data_stud.drop('obr_uchr_name', axis=1, inplace=True)
    data_stud.drop('data_dok_obr', axis=1, inplace=True)
    data_stud.drop('Направление', axis=1, inplace=True)

    # Добавление колонки оценок в университете в основную таблицу
    data_stud.index = data_stud['proid']
    data_stud['ball1'] = arr

    # Удаление записей, где оценоки в университете не указаны
    data_stud = data_stud.query('ball1 != "NaN"')

    # Сохранение в файл
    # data_stud.to_csv('data_stud_test.csv', sep=';')
    return data_stud


def make_model(data_stud: pd.DataFrame):
    # data_stud = pd.read_csv('data_stud_test.csv', delimiter=';', encoding='windows-1251')
    print(data_stud.shape[0])

    learn_barier = int(len(data_stud) / 2)
    # X = np.array(data_stud[['obsh', 'ball1', 'kurs', 'ex_srball', 'att_srball']][0:-1000])
    # X = np.array(data_stud[['ball1', 'ex_srball', 'dkb', 'fac', 'a_god_priema']][0:learn_barier])
    X = np.array(data_stud[['ball1', 'ex_srball']][0:learn_barier])
    y = np.array(data_stud[['razdel']][0:learn_barier])
    testX = np.array(data_stud[['ball1', 'ex_srball']][learn_barier:])
    testY = np.array(data_stud[['razdel']][learn_barier:])

    clf = MLPClassifier(solver='lbfgs', alpha=1e-9, verbose=False, activation='relu',
                        hidden_layer_sizes=(20, 20), max_iter=200, random_state=1)
    clf.fit(X, y)

    count0 = 0
    count1 = 0
    for i in range(len(testX)):
        # Печать отчисленных предсказанием или по факту
        if testY[i] == 0 or clf.predict([testX[i]]) == 0:
            print(testX[i], testY[i], clf.predict([testX[i]]))
            # testX[i], testY[i], clf.predict([testX[i]]))
        # Подсчёт отчисленных и не отчисленных моделью
        if clf.predict([testX[i]]) == 0:
            count0 += 1
        else:
            count1 += 1

    print('Total not drop students: ', count1)
    print('Total drop students: ', count0)

    ''''# ones predicts
    print(clf.predict([[5.0, 70]]))
    print(clf.predict([[4.5, 50]]))
    print(clf.predict([[4.0, 60]]))
    print(clf.predict([[3.5, 40]]))
    print(clf.predict([[3.0, 90]]))
    print()
    print(clf.predict([[2.9, 70]]))
    print(clf.predict([[2.5, 60]]))
    print(clf.predict([[2.0, 50]]))
    print(clf.predict([[1.5, 40]]))
    '''
    return clf


if __name__ == '__main__':
    # Пути файлам датасета из консоли
    # students, marks = argv
    students, marks = 'Студенты.csv', 'Оценки.csv'
    # Преобразование данных
    data = format_dataset(students, marks)
    # Создание модели и небольшое ручное тестирование предсказаний
    model = make_model(data)

    # Сохранение модели на диск
    filename = 'clf_model'
    pickle.dump(model, open(filename, 'wb'))



