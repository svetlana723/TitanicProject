"""
Проект Титаник
Этап 1: Разведочный анализ данных (EDA)
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns


DF_TRAINING_PATH = "./data/train.csv"
DF_TEST_PATH = "./data/test.csv"

def print_break_line(text=""):
    len_text = len(text)
    print()
    if len_text > 0:
        print('='*25 + ' ' + text + ' ' + '='*(113 - len_text))
    else:
        print('='*140)
    print()


def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def print_main_stats(df: pd.DataFrame):
    print(f"Число строк: {len(df)}")
    print(f"Число колонок: {df.shape[1]}")
    print()
    print(df.info())
    print()
    print("Первые 5 строк:")
    print(df.head())
    print()
    print("Последние 3 строки:")
    print(df.tail(3))
    print()

def print_missing(df: pd.DataFrame):
    print(df.isna().sum())
    print()
    print(f"Общее количество пропусков: {df.isna().sum().sum()}")
    print()

def print_basic_analysis(df: pd.DataFrame):
    
    print("Сколько всего пассажиров?")
    print(len(df))

    print("Сколько мужчин и женщин?")
    print(f"Мужчин: {len(df[df['Sex'] == 'male'])}")
    print(f"Женщин: {len(df[df['Sex'] == 'female'])}")
    
    print("Кто выживал чаще?")
    print(f"Мужчины: {len(df[(df['Sex'] == 'male') & (df['Survived'] == 1)])}")
    print(f"Женщины: {len(df[(df['Sex'] == 'female') & (df['Survived'] == 1)])}")

    print("Влиял ли класс на выживаемость?")
    print(df.groupby('Pclass').agg({'Survived': ['sum', 'count']}))

    print("Есть ли разница между портами посадки?")
    print(df.groupby('Embarked').agg({'Survived': ['sum', 'count']}))

    print()

def show_plots(df: pd.DataFrame):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # 2 строки, 2 колонки
    fig.suptitle('Анализ данных Титаника')

    # График 1: распределение выживших
    df['Survived'].value_counts().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Распределение выживших')
    axes[0, 0].set_xlabel('0 - погиб, 1 - выжил')
    axes[0, 0].tick_params(axis='x', rotation=0)
    axes[0, 0].set_ylabel('Количество')
    
    # График 2: выживаемость по полу
    pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Выживаемость по полу')
    axes[0, 1].set_xlabel('Sex')
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].set_ylabel('Количество')
    
    # График 3: выживаемость по классам
    pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Выживаемость по классам')
    axes[1, 0].set_xlabel('Pclass')
    axes[1, 0].tick_params(axis='x', rotation=0)
    axes[1, 0].set_ylabel('Количество')
    
    # График 4: возраст
    axes[1, 1].hist([df[df['Survived']==1]['Age'].dropna(), 
                     df[df['Survived']==0]['Age'].dropna()], 
                    bins=30, alpha=0.7, label=['Выжили', 'Погибли'])
    axes[1, 1].set_title('Распределение возраста')
    axes[1, 1].set_xlabel('Возраст')
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].set_ylabel('Количество')
    axes[1, 1].legend()
    
    plt.tight_layout()  # чтобы графики не налезали друг на друга

    fig.savefig('plots.png')  # <- сохраняем всё окно
    plt.show()

def main():
    df_train, df_test = load_data(DF_TRAINING_PATH, DF_TEST_PATH)

    print_break_line("Training Dataset")
    print_main_stats(df_train)

    print_break_line("Test Dataset")
    print_main_stats(df_test)

    print_break_line("Training Dataset Missing")
    print_missing(df_train)

    print_break_line("Test Dataset Missing")
    print_missing(df_test)

    print_break_line("Training Dataset Research")
    print_basic_analysis(df_train)

    show_plots(df_train)

if __name__ == "__main__":
    main()