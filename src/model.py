import pandas as pd
import statsmodels.formula.api as smf

# 関数の定義
def fit_simple_regression(data, x_col:str, y_col:str):
    """
    単回帰モデルを学習する
    """
    formula = f"{y_col} ~ {x_col}"
    model = smf.ols(formula=formula, data=data).fit()
    return model

def dict_metrics(model):
    """
    モデルの主要指標を辞書で返す
    """
    x_name = model.model.exog_names[1] #切片を除いた説明変数名
    return {
        "intercept": model.params["Intercept"],
        "coef": model.params[x_name],
        "p_value": model.pvalues[x_name],
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
    }

def print_metrics(metrics: dict):
    print("切片:", round(metrics["intercept"], 3))
    print("係数:", round(metrics["coef"], 3))
    print("p値:", round(metrics["p_value"], 6))
    print("決定係数:", round(metrics["r_squared"], 3))
    print("調整済み決定係数:", round(metrics["adj_r_squared"], 3))

def predict(model, new_df):
    """
    新しいデータに対する予測値を返す
    """
    return model.predict(new_df)

# 使用例
if __name__ == "__main__": 
    df = pd.read_csv('../data/day.csv')
    model = fit_simple_regression(df, "temp", "cnt")
    print_metrics(dict_metrics(model))
    new_data = pd.DataFrame({"temp":[0.2]})
    print(predict(model, new_data))