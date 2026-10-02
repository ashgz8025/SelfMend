"""Test GX 1.23 full validation flow."""
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
)
import pandas as pd

df = pd.DataFrame({
    "Quantity": [1.0, 2.0, None, 4.0, 5.0],
    "Item": ["Coffee", "Tea", "Coffee", "Juice", "ERROR"],
    "Price Per Unit": [3.5, 2.5, 3.5, 3.0, 3.0],
})

ctx = gx.get_context(mode="ephemeral")
ds  = ctx.data_sources.add_pandas("ds")
da  = ds.add_dataframe_asset("asset")
bd  = da.add_batch_definition_whole_dataframe("bd")

suite = ctx.suites.add(gx.ExpectationSuite("my_suite"))
suite.add_expectation(ExpectColumnToExist(column="Quantity"))
suite.add_expectation(ExpectColumnValuesToBeBetween(
    column="Quantity", min_value=1, max_value=5, mostly=0.7
))
suite.add_expectation(ExpectColumnValuesToBeInSet(
    column="Item",
    value_set=["Coffee", "Tea", "Juice", "Cake", "Sandwich", "Cookie", "Smoothie", "Salad"],
    mostly=0.9
))

vd = ctx.validation_definitions.add(
    gx.ValidationDefinition(name="vd", data=bd, suite=suite)
)

result = vd.run(batch_parameters={"dataframe": df})
print("Validation success:", result.success)
for r in result.results:
    status = "✓" if r.success else "✗"
    print(f"  {status} {r.expectation_config.type} → {r.success}")
    if not r.success:
        print(f"    observed: {r.result}")
