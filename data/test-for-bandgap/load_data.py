# load data from mp
import csv
import pandas as pd
from pymatgen import MPRester

if __name__ == '__main__':
    MP_KEY = 'cRelC34nhXp1wf5H'
    mp_dr = MPRester(MP_KEY)
    # query
    criteria = {
        "nelements": {"$gte": 1}, 
        "band_gap": {"$gt": 0 }, 
        "e_above_hull": {"$lt": 0.000001 },
    }
    # property 
    properties = [
        'material_id',
        'band_gap',
        'cif'
    ]
    data = mp_dr.query(criteria=criteria, properties=properties)
    list_id_and_band_gap = [[value['material_id'].strip("mp-""vc-"), value['band_gap']] for value in data]
    with open("data.csv", "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(list_id_and_band_gap)

    for value in data:
        with open("{}.cif".format(value['material_id'].strip("mp-""vc-")), mode='w') as f:
            f.write(value['cif'])
