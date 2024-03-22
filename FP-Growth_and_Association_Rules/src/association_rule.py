from itertools import combinations
import numpy as np
import pandas as pd



def antecedent_support(sAC, sA, sC):
    return sA
def consequent_support(sAC, sA, sC):
    return sC
def support(sAC, sA, sC):
    return sAC
def confidence(sAC, sA, sC):
    return sAC / sA

def lift(sAC, sA, sC):
    return confidence(sAC, sA, sC) / sC

def leverage(sAC, sA, sC):
    return support(sAC, sA, sC) - sA * sC

def conviction(sAC, sA, sC):
    confidence = sAC / sA
    conviction = np.empty(confidence.shape, dtype=float)
    if len(conviction.shape) == 0:
        conviction = conviction[np.newaxis]
        confidence = confidence[np.newaxis]
        sAC = sAC[np.newaxis]
        sA = sA[np.newaxis]
        sC = sC[np.newaxis]
    conviction[:] = np.inf
    conviction[confidence < 1.0] = (1.0 - sC[confidence < 1.0]) / (
        1.0 - confidence[confidence < 1.0]
    )
    return conviction

def zhangs_metric(sAC, sA, sC):
    denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
    numerator = leverage(sAC, sA, sC)
    with np.errstate(divide="ignore", invalid="ignore"):
        # ignoring the divide by 0 warning since it is addressed in the below np.where
        zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)
    return zhangs_metric

metrics_dict: dict = {
    "antecedent support": antecedent_support,
    "consequent support": consequent_support,
    "support": support,
    "confidence": confidence,
    "lift": lift,
    "leverage": leverage,
    "conviction": conviction,
    "zhangs_metric": zhangs_metric
}

def association_rules_custom(dataframe: pd.DataFrame, metric: str='support', threshold:float = 0.02):

    """
    Определяет ассоциативные правила по заданной метрике
    Расчитывает значение остальных метрик

    Arguments
    --------
        :dataframe: данные с выхода FP-Growth со столбцами ['itemsets', 'support']
        :metric: название метрики для построения правил
        :threshold: пороговое значение метрики для формирования ассоциативного правила
    
    Returns:
    -------
        :association_rules_df: список ассоциативных правил с метриками
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Ожидался датафрэйм pandas, а получено {}".format(type(dataframe)))
    if not all(column in dataframe.columns for column in ['itemsets', 'support']):
        raise ValueError("Колонки должны быть ['itemsets', 'support'], а тут {}".format(dataframe.columns))
    if not metric in metrics_dict:
        raise ValueError("Неизвестная метрика {}".format(metric))

    itemset = dataframe['itemsets']
    support_values = dataframe['support']
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(itemset), support_values))

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    for itemset in frequent_items_dict:
        sAC = frequent_items_dict[itemset]
        # пропускаем первый элемент
        for indx in range(1, len(itemset)):
            # ищем другие комбинации с объектами
            for combination in combinations(itemset, r=indx):
                A = frozenset(combination)
                C = itemset.difference(A)

                sA = frequent_items_dict[A]
                sC = frequent_items_dict[C]
                score = metrics_dict['lift'](sAC, sA, sC)
                if score >= threshold:
                    rule_antecedents.append(A)
                    rule_consequents.append(C)
                    rule_supports.append([sAC, sA, sC])

    # Считаем остальные метрики
    rule_supports = np.array(rule_supports).astype(float)
    association_rules_df = pd.DataFrame(
        data=list(zip(rule_antecedents, rule_consequents)),
        columns=["antecedents", "consequents"],
    )

    sAC = rule_supports[..., 0]
    sA = rule_supports[..., 1]
    sC = rule_supports[..., 2]
    for m in metrics_dict:
        association_rules_df[m] = metrics_dict[m](sAC, sA, sC)
    return association_rules_df
