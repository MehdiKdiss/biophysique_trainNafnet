from cellpose import models
print(dir(models))
try:
    model = models.Cellpose(model_type='cyto')
    print("Cellpose class works")
except Exception as e:
    print(f"Cellpose class failed: {e}")

try:
    model = models.CellposeModel(model_type='cyto')
    print("CellposeModel class works")
except Exception as e:
    print(f"CellposeModel class failed: {e}")
