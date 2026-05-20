# モデルを構築するための関数

def fit_multiple_regression(
    df,
    target,
    numeric_features,
    categorical_features=None,
    quadratic_features=None,
    verbose=True):
    '''
    数値説明変数・カテゴリ変数・2次項を指定して重回帰モデルを作成する関数。

    Parameters
    ----------
    df : pandas.DataFrame
        分析対象のデータフレーム。
    target : str
        目的変数の列名。
    numeric_features : list of str
        数値の説明変数の列名リスト。
    categorical_features : list of str, optional
        カテゴリ変数の列名リスト。
    quadratic_features : list of str, optional
        2次項として追加したい数値変数の列名リスト。
        例: ['temp', 'hum'] を指定すると I(temp ** 2), I(hum ** 2) を追加する。
    verbose : bool, default True
        True の場合、使用する回帰式を表示する。

    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        statsmodels の回帰分析結果。
    '''
    import statsmodels.formula.api as smf

    try:
        if categorical_features is None:
            categorical_features = []

        if quadratic_features is None:
            quadratic_features = []

        if numeric_features is None:
            numeric_features = []

        # 説明変数が1つもない場合
        if (
            len(numeric_features) == 0
            and len(categorical_features) == 0
            and len(quadratic_features) == 0
        ):
            raise ValueError('説明変数が指定されていません。')

        # 使用する列が df に存在するか確認
        required_columns = (
            [target]
            + numeric_features
            + categorical_features
            + quadratic_features
        )

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f'データフレームに存在しない列があります: {missing_columns}')

        # 数値変数
        numeric_part = numeric_features

        # カテゴリ変数
        categorical_part = [f'C({feature})' for feature in categorical_features]

        # 2次項
        quadratic_part = [f'I({feature} ** 2)' for feature in quadratic_features]

        # 説明変数を結合
        all_features = numeric_part + categorical_part + quadratic_part

        # 回帰式を作成
        formula = target + ' ~ ' + ' + '.join(all_features)

        if verbose:
            print('使用する回帰式:')
            print(formula)

        # モデル作成
        model = smf.ols(formula=formula, data=df).fit()

        return model

    except Exception as e:
        print('ERROR:')
        print(e)
        return None


#------------------------------------------------------------------------------------------

def verif_model(model, numeric_features, categorical_features=None, quadratic_features=None, model_name=None):
    '''
    構築した回帰モデルを検証するための関数

    Parameters
    ----------
    model : statsmodels の回帰モデル
        smf.ols(...).fit() で作成したモデル

    numeric_features : list
        数値の説明変数のリスト
        例: ['temp', 'hum', 'windspeed']

    categorical_features : list, optional
        カテゴリー変数のリスト
        例: ['season', 'weathersit']
        
    quadratic_features : list, optional
        2次項を組み込んだ変数のリスト
        例: ['temp', 'hum']

    model_name : str

    Returns
    -------
    df_results_params : pandas.DataFrame
        回帰係数、p値、標準誤差、t値をまとめたデータフレーム

    df_model_results : pandas.DataFrame
        R^2、Adj. R^2、AIC、BICをまとめたデータフレーム
    '''

    import pandas as pd

    if categorical_features is None:
        categorical_features = []

    if quadratic_features is None:
        quadratic_features = []

    results = []

    # 数値変数の結果
    for feature in numeric_features:
        if feature in model.params.index:
            results.append({
                '変数': feature,
                '種類': '数値変数',
                '係数': model.params[feature],
                'p値': model.pvalues[feature],
                '標準誤差': model.bse[feature],
                't値': model.tvalues[feature]
            })
        else:
            print(f'注意: {feature} はモデル内に見つかりません。')

    # カテゴリー変数の結果
    for feature in categorical_features:
        category_params = [
            param for param in model.params.index
            if param.startswith(f'C({feature})')
        ]

        if len(category_params) == 0:
            print(f'注意: {feature} に対応するカテゴリー係数が見つかりません。')
            continue

        for param in category_params:
            results.append({
                '変数': param,
                '種類': f'カテゴリー変数: {feature}',
                '係数': model.params[param],
                'p値': model.pvalues[param],
                '標準誤差': model.bse[param],
                't値': model.tvalues[param]
            })

    # 2次項の結果
    for feature in quadratic_features:
        quadratic_params = [
            param for param in model.params.index
            if param.startswith(f'I({feature} ** 2)') or param.startswith(f'I({feature}**2)')
        ]

        if len(quadratic_params) == 0:
            print(f'注意: {feature} に対応する2次項の係数が見つかりません。')
            continue

        for param in quadratic_params:
            results.append({
                '変数': param,
                '種類': f'{feature} の2次項',
                '係数': model.params[param],
                'p値': model.pvalues[param],
                '標準誤差': model.bse[param],
                't値': model.tvalues[param]
            })

    df_results_params = pd.DataFrame(results)

    # モデル全体の評価指標
    model_results = [{
        '決定係数 R^2': model.rsquared,
        '自由度調整済み決定係数 Adj. R^2': model.rsquared_adj,
        'AIC': model.aic,
        'BIC': model.bic
    }]

    df_model_results = pd.DataFrame(model_results)
    if model_name is None:
        df_model_results.index = [f'{model}']
    else:
        df_model_results.index = [model_name]

    return df_results_params, df_model_results

#------------------------------------------------------------------------------------

# 説明変数ごとの散布図に単回帰直線を重ねる
def scatter_and_model(data, numeric_features, target):
    '''
    数値説明変数ごとに、目的変数との散布図を作成し、
    単回帰直線を重ねて表示する関数。

    Parameters
    ----------
    data : pandas.DataFrame
        分析対象のデータフレーム。
    numeric_features : list of str
        散布図を作成する数値説明変数名のリスト。
    target : str
        目的変数名。

    Returns
    -------
    None
        各説明変数について散布図と回帰直線を表示する。
    '''
    from matplotlib import pyplot as plt
    import seaborn as sns

    try:
        for feature in numeric_features:
            sns.lmplot(
                x=feature,
                y=target,
                data=data,
                scatter_kws={'color': 'gray'},
                line_kws={'color': 'black'},
                ci=None,
                height=6,
                aspect=2
            )

            plt.xlabel(feature)
            plt.ylabel(target)
            plt.title(f'{feature}と{target}の散布図')
            plt.show()

    except Exception as e:
        print('ERROR :', e)

#------------------------------------------------------------------------------

# 残差と予測値を取得し、図示する関数
def resid_and_predict(models: dict):
    '''
    statsmodelsで作成した回帰モデルについて、
    予測値と残差の関係を可視化する関数。

    Parameters
    ----------
    models : dict
        モデル名をキー、statsmodelsの回帰結果オブジェクトを値とする辞書。
        例：
        {
            '単回帰モデル_temp': temp_model,
            '重回帰モデル': mul_model
        }

    Returns
    -------
    None
        各モデルについて、予測値と残差の散布図を表示する。
    '''
    import seaborn as sns
    from matplotlib import pyplot as plt

    try:
        for model_name, model in models.items():
            # 残差の取得
            resid = model.resid

            # 当てはめ値
            fitted = model.fittedvalues

            plt.figure(figsize=(8, 5))

            # 残差と予測値の散布図
            sns.scatterplot(
                x=fitted,
                y=resid,
                color='blue',
                alpha=0.3
            )

            # LOWESSによる傾向線
            sns.regplot(
                x=fitted,
                y=resid,
                scatter=False,
                lowess=True,
                color='black'
            )

            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('予測値')
            plt.ylabel('残差')
            plt.title(f'残差 vs 予測値（{model_name}）')
            plt.show()

    except Exception as e:
        print('ERROR :\n', e)

#-----------------------------------------------------------------
# MAE,RMSE,R2によるモデル評価
def mae_rmse_r2(models, test_df):
    '''
    statsmodelsで作成した回帰モデルを、
    テストデータに対してMAE, RMSE, R2で評価する関数。

    Parameters
    ----------
    models : dict of model
        構築したモデルのリスト
        Ex : {'モデル名', model , ...}

    test_df : pandas.DataFrame
        テストデータ

    target : str, default 'cnt'
        目的変数名

    Returns
    -------
    results_df : pandas.DataFrame
        MAE, RMSE, R2をまとめたデータフレーム
    '''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    import pandas as pd
    try:
        results = []
        for model_name, model in models.items():
            # 実測値
            y_true = test_df['cnt']
            # 予測値
            y_pred = model.predict(test_df)
        
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
        
            # データを格納
            results.append ({
                'モデル名': f'{model_name}',
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2})
        
        # データフレームにまとめる
        results_df = pd.DataFrame(results)
        
        return results_df
    except Exception as e:
        print('ERROR :\n', e)

# テストデータに対して予測値と残差をプロットする関数
def test_pred_resid(models: dict, test_df):
    from matplotlib import pyplot as plt
    import seaborn as sns

    try:
        for model_name, model in models.items():
            # 実測値
            y_true = test_df['cnt']

            # 予測値
            y_pred = model.predict(test_df)

            # 残差
            resid = y_true - y_pred

            plt.figure(figsize=(8, 5))

            sns.scatterplot(
                x=y_pred,
                y=resid,
                color='blue',
                alpha=0.3
            )

            sns.regplot(
                x=y_pred,
                y=resid,
                scatter=False,
                lowess=True,
                color='black'
            )

            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('予測値')
            plt.ylabel('残差（実測値 - 予測値）')
            plt.title(f'予測値と残差（{model_name}）')
            plt.show()

    except Exception as e:
        print('ERROR :\n', e)