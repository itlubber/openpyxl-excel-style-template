# python公共模块之openpyxl修改excel文件样式

![python公共模块之openpyxl修改excel文件样式](https://itlubber.art/upload/2022/10/2022102501.png)

## 概述


> 在给客户做项目的过程中可能需要交付 `excel` 文档，每次都要花费一定的时间来修改格式，感觉略烦，故而准备直接用 `python` 的 `openpyxl` 库来修改 `excel` 的格式，省心省力，瞬间世界清爽 QaQ
> 
> 本文相关代码和文章已同步至 `github` 和 `微信公众号` ，各位大佬按需取用。
> 
> 代码开源地址：[`https://github.com/itlubber/openpyxl-excel-style-template/tree/main`](https://github.com/itlubber/openpyxl-excel-style-template/tree/main)
>
> 博客网站推文：[`https://itlubber.art/archives/openpyxl-excel-style-template`](https://itlubber.art/archives/openpyxl-excel-style-template)
> 
> 微信公众号推文：[`暂无发布`](https://mp.weixin.qq.com/s/ozvbv-ToHB4gQe5LKw4PXQ)


## 代码结构

![代码结构](https://itlubber.art/upload/2022/10/2022102503.png)

## 代码说明

> `format_bins` : 特定场景下的特定需求下定制函数，可忽略
> 
> `feature_bin_stats` : 特定场景下的特定需求下定制函数，可忽略
> 
> `plot_bin` : 特定场景下的特定需求下定制函数，可忽略
> 
> `cal_psi` : 特定场景下的特定需求下定制函数，可忽略
> 
> `itlubber_border` : 边框格式设置函数
> 
> `render_excel` : excel 样式渲染函数


## 食用方式

```python
data = sc.germancredit()

# 测试数据
data["target"] = data["creditability"].replace({'good':0,'bad':1})
data["credit.amount"].loc[0] = np.nan
data["status.of.existing.checking.account"].loc[0] = np.nan
data["test_a"] = 0.
data["test_a"].loc[0] = np.nan
data["test_b"] = ""
data["test_b"].loc[0] = np.nan

train, test = train_test_split(data, test_size=0.3,)

target = "target"
cols = ["test_a", "test_b", "status.of.existing.checking.account", "credit.amount"]

combiner = toad.transform.Combiner()
combiner.fit(data[cols + [target]], target, empty_separate=True, method="chi", min_samples=0.2)

# 保存结果至 EXCEL 文件
output_excel_name = "指标有效性验证.xlsx"
output_sheet_name = "指标有效性"
tables = {}
merge_row_number = []

for feature in cols:
    table = feature_bin_stats(train, feature, feature_dict=feature_dict, rules={})
    df_psi = cal_psi(train[[feature, target]], test[[feature, target]], feature, combiner=combiner)
    
    table = table.merge(df_psi, on="分箱", how="left")
    
    feature_bin = combiner.export()[feature]
    feature_bin_dict = format_bins(np.array(feature_bin))
    table["分箱"] = table["分箱"].map(feature_bin_dict)
    
    # plot_bin(table, show_na=True)
    merge_row_number.append(len(table))
    tables[feature] = table

merge_row_number = np.cumsum(merge_row_number).tolist()
feature_table = pd.concat(tables, ignore_index=True).round(6)
feature_table["分档WOE值"] = feature_table["分档WOE值"].fillna(np.inf)
feature_table.to_excel(output_excel_name, sheet_name=output_sheet_name, index=False, header=True, startcol=0, startrow=0)

render_excel(output_excel_name, sheet_name=output_sheet_name, conditional_columns=["J", "N"], freeze="D2", merge_rows=merge_row_number, percent_columns=[5, 7, 9, 10])
render_excel("变量字典及字段解释.xlsx")
```


## 渲染结果

![正常使用样式](https://itlubber.art/upload/2022/10/2022102502.png)


![特定场景下的渲染样式](https://itlubber.art/upload/2022/10/2022102501.png)


## 代码说明

具体说明请查阅微信公众或博客文章 QaQ
