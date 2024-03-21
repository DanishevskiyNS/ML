import numpy as np
import pandas as pd

def get_unique_elements_and_mappings(transactions: np.ndarray):
    """
    Получает список уникальных объектов и мэппит их с их индексами
    
    Arguments:
    -----
        :transactions: массив всех транзакций
    Returns:
    -------
        :columns: список всех уникальных предметов в отсортированном виде
        :items_mapping: мапа 
    """
    columns = np.unique(transactions)
    columns = columns[columns != 'nan']
    print(columns.shape)
    columns.sort()
    items_mapping = {}
    for col_indx, item in enumerate(columns):
        items_mapping[item] = col_indx
    return columns, items_mapping

def encode_transactions(transactions: np.ndarray, as_frame:bool=True) -> np.ndarray | pd.DataFrame:
    """
    Кодирует каждую транзакцию в булевый массив
    
    Arguments:
    -----
        :transactions: массив всех транзакций
        :as_frame: позволяет возвращать в виде pd.DataFrame
    
    Returns:
    -------
        :encoded_transaction: закодированные транзакции
    
    Пример:
    ------
    ```
    transaction = [
        [Продукт 1, Продукт 3, Продукт 5],
        [Продукт 1, Продукт 2, Продукт 3, Продукт 4, Продукт 5],
        [Продукт 5]
    ]
    columns = [Продукт 1, Продукт 2, Продукт 3, Продукт 4, Продукт 5]

    encoded_transactions = [
        [True, False, True, False, True],
        [True, True, True, True, True],
        [False, False, False, False, True]
    ]
    ```
    """

    columns, _ = get_unique_elements_and_mappings(transactions)
    
    encoded_transactions = np.zeros((transactions.shape[0], len(columns)), dtype=bool)
    for item_id, item in enumerate(columns):
        transaction_id = np.where(transactions == item)[0]
        encoded_transactions[transaction_id, item_id] = True
    
    if as_frame:
        return pd.DataFrame(encoded_transactions, columns=columns)
    return encoded_transactions
