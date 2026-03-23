# Titanic Survival Prediction

## Описание проекта
Предсказание выживаемости пассажиров Титаника с использованием машинного обучения.

## Структура проекта
- `src/eda.py` — разведочный анализ данных
- `notebooks/fe.ipynb` — подготовка данных и создание признаков
- `notebooks/modeling.ipynb` — обучение и сравнение моделей
- `data/` — исходные и обработанные данные, файл с предсказаниями

## Созданные признаки
- Title (титул из имени)
- FamilySize (размер семьи)
- IsAlone (одинокий)
- FarePerPerson (стоимость билета на человека)
- AgeGroup (возрастная группа)
- HasCabin (наличие каюты)
- Deck (палуба)

## Результаты
Модель - Accuracy

Logistic Regression - 0.8212

Random Forest - 0.7877

## Выводы
Самые важные признаки для предсказания:
1. Title_Mrs (титул "миссис")
2. Sex (пол)
3. FarePerPerson (стоимость билета на человека)

## Как запустить
1. Клонировать репозиторий
2. Установить зависимости: `pip install -r requirements.txt`
3. Запустить src/eda.py, notebooks/fe.ipynb, notebooks/modelling.ipynb
