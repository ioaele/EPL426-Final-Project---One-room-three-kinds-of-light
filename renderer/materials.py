import numpy as np

KD = 0            
KS = 3   
M = 6             
KR = 7        
KT = 8        
IOR = 9           
ALPHA = 10        
GLOSS = 11        
EMISSION = 12    
CHECKER = 15     
CHECKER_SIZE = 16 
KD2 = 17         
MAT_SIZE = 20

DEFAULTS = {
    "kd": [0.7, 0.7, 0.7],
    "ks": 0.0,
    "m": 32.0,
    "kr": 0.0,
    "kt": 0.0,
    "ior": 1.5,
    "alpha": 1.0,
    "gloss": 0.0,
    "emission": [0.0, 0.0, 0.0],
}


def _rgb(value):
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return np.repeat(a, 3)
    if a.size != 3:
        raise ValueError(f"expected one number or 3 numbers, got {value!r}")
    return a


def material_row(desc):
    desc = desc or {}
    row = np.zeros(MAT_SIZE, dtype=np.float64)
    row[KD:KD + 3] = _rgb(desc.get("kd", DEFAULTS["kd"]))
    row[KS:KS + 3] = _rgb(desc.get("ks", DEFAULTS["ks"]))
    row[M] = float(desc.get("m", DEFAULTS["m"]))
    row[KR] = float(desc.get("kr", DEFAULTS["kr"]))
    row[KT] = float(desc.get("kt", DEFAULTS["kt"]))
    row[IOR] = float(desc.get("ior", DEFAULTS["ior"]))
    row[ALPHA] = float(desc.get("alpha", DEFAULTS["alpha"]))
    row[GLOSS] = float(desc.get("gloss", DEFAULTS["gloss"]))
    row[EMISSION:EMISSION + 3] = _rgb(desc.get("emission", DEFAULTS["emission"]))
    checker = desc.get("checker")
    if checker:
        row[CHECKER] = 1.0
        row[CHECKER_SIZE] = float(checker["size"])
        row[KD2:KD2 + 3] = _rgb(checker["kd2"])
    return row


class MaterialTable:

    def __init__(self):
        self._rows = []
        self._index = {}

    def index_of(self, desc):
        row = material_row(desc)
        key = row.tobytes()
        if key not in self._index:
            self._index[key] = len(self._rows)
            self._rows.append(row)
        return self._index[key]

    def to_array(self):
        if not self._rows:
            self.index_of({})
        return np.vstack(self._rows)


def load_materials(material_descs):
    """Build the material array from a list of material dicts.

    Returns (materials array, list of row indices, one per input dict).
    """
    table = MaterialTable()
    ids = [table.index_of(d) for d in material_descs]
    return table.to_array(), ids
